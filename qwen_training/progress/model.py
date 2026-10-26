# ruff: noqa: RUF002, RUF003
"""流式进度模型：Qwen3.5 底座 + [STEP]/[STG]/[CNT]/[P] readout（proposal_progress 3.2–3.5、第 4 节）。

- 视觉塔、词嵌入冻结；语言模型线性层 LoRA；GDN 的 in_proj_a/b、A_log、dt_bias 全量训练。
- 新 token 的 embedding、计数阈值和 MLP 头全量训练，与 LoRA 一起保存。
"""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

from qwen_training.datasets.rmbench.labels import MAX_COUNT
from qwen_training.progress.stream import CNT, FRAME, PREFIX, STG, P, StreamState, Tokens, forward_chunk

STEP_TOKEN, STG_TOKEN, CNT_TOKEN, P_TOKEN = range(4)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj")
GATE_PARAMS = ("in_proj_a", "in_proj_b", "A_log", "dt_bias")


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, x):
        # 在 autocast 下计算，不把输入转成 fp32：否则每层每个投影都多存一份 fp32 激活。
        return self.base(x) + self.up(self.down(x)) * self.scale


class ProgressModel(nn.Module):
    def __init__(self, backbone, image_processor, tokenizer, cfg):
        super().__init__()
        self.backbone, self.image_processor, self.tokenizer, self.cfg = backbone, image_processor, tokenizer, cfg
        backbone.requires_grad_(False)
        language = backbone.model.language_model
        width, hidden = backbone.config.text_config.hidden_size, cfg.head_hidden
        device = language.embed_tokens.weight.device
        scale = language.embed_tokens.weight.float().std().item()
        self.special = nn.Parameter(torch.randn(4, width, device=device) * scale)

        def head(inputs, outputs):
            return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs)).to(device)

        # 上一步的计数概率回传：Σ_j MLP([h_STEP_j; P_{t-1}(n_j ≥ 1..R)]) 加到这一步 [CNT] 的输入上；
        # 末层零初始化，开始时等于不回传。
        self.count_feedback = head(width + MAX_COUNT, width)
        nn.init.zeros_(self.count_feedback[-1].weight)
        nn.init.zeros_(self.count_feedback[-1].bias)

        self.stage_head, self.none_head = head(2 * width, 1), head(width, 1)
        self.count_head, self.progress_head = head(2 * width, 1), head(width, cfg.progress_bins)
        # 累积计数：P(第 j 项做了至少 r 次) = σ(g_j − θ_r)，θ_1 < … < θ_R 由首项加 softplus 间隔保证递增，初始为 0…R−1。
        self.count_first = nn.Parameter(torch.zeros(1, device=device))
        self.count_gaps = nn.Parameter(torch.full((MAX_COUNT - 1,), math.log(math.e - 1), device=device))
        self.register_buffer("bins", torch.linspace(0, 1, cfg.progress_bins, device=device), persistent=False)

        for layer in language.layers:
            for parent in layer.modules():
                for name, child in list(parent.named_children()):
                    if name in LORA_TARGETS and isinstance(child, nn.Linear):
                        setattr(parent, name, LoRALinear(child, cfg.lora_rank, cfg.lora_alpha))
        self.gates = []   # (参数, 初始值)；训练时加 L2，让它们不远离初始值
        for name, param in language.named_parameters():
            if any(part in GATE_PARAMS for part in name.split(".")):
                param.data = param.data.float()
                param.requires_grad_(True)
                self.gates.append((param, param.detach().clone()))

    @classmethod
    def from_pretrained(cls, path, cfg, device="cuda"):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(path)
        return cls(backbone, processor.image_processor, processor.tokenizer, cfg)

    def load_trainable(self, weights):
        """checkpoint 只保存可训练参数；结构变化时拒绝静默漏载。"""
        expected = {name for name, param in self.named_parameters() if param.requires_grad}
        missing, unexpected = expected - weights.keys(), weights.keys() - expected
        if missing or unexpected:
            raise ValueError(f"checkpoint 参数不匹配: 缺少 {sorted(missing)}, 多余 {sorted(unexpected)}")
        self.load_state_dict(weights, strict=False)

    @property
    def language(self):
        return self.backbone.model.language_model

    @torch.no_grad()
    def encode_images(self, images):
        """RGB uint8 数组列表 → [(特征 [n, d], (h, w))]，h、w 为合并后的网格。"""
        inputs = self.image_processor(images=images, return_tensors="pt")
        grids = inputs["image_grid_thw"].to(self.special.device)
        visual = self.backbone.model.visual
        features = self.backbone.model.get_image_features(
            inputs["pixel_values"].to(self.special.device, visual.dtype), grids).pooler_output
        merge = visual.spatial_merge_size
        return [(feature, (int(grid[1]) // merge, int(grid[2]) // merge))
                for feature, grid in zip(features, grids, strict=True)]

    def _forward(self, state, items, frames=None, keyframes=()):
        """items: [(kind, 内容)]，内容是文本、特殊 token 序号、embedding [1, d] 或 (图像特征, (h, w))。

        拼成一段接到缓存后前向，返回每个 item 对应的 hidden。图像前后加 vision_start、vision_end；
        位置按 Qwen3.5 的 get_rope_index：文本三轴相同，图像为 (起点, 起点 + 行, 起点 + 列)。
        frames：每个 item 所在的帧号，默认都是 state.frame（一段多步并行时逐个给出）；keyframes 中的帧标为关键帧。
        """
        device, embed, config = self.special.device, self.language.embed_tokens, self.backbone.config
        marks = embed(torch.tensor([config.vision_start_token_id, config.vision_end_token_id], device=device))
        frames = frames or [state.frame] * len(items)
        embeds, positions, kinds, frame_ids, lengths = [], [], [], [], []
        for (kind, content), frame in zip(items, frames, strict=True):
            if isinstance(content, str):
                ids = self.tokenizer(content, add_special_tokens=False)["input_ids"]
                parts = [(embed(torch.tensor(ids, device=device)), None)]
            elif isinstance(content, int):
                parts = [(self.special[content : content + 1], None)]
            elif isinstance(content, torch.Tensor):
                parts = [(content, None)]
            else:
                parts = [(marks[:1], None), content, (marks[1:], None)]
            for part, grid in parts:
                start = state.next_position
                if grid is None:
                    positions.append(torch.arange(start, start + len(part)).expand(3, -1))
                    state.next_position += len(part)
                else:
                    height, width = grid
                    positions.append(torch.stack([torch.full((height * width,), start),
                                                  torch.arange(start, start + height).repeat_interleave(width),
                                                  torch.arange(start, start + width).repeat(height)]))
                    state.next_position += max(height, width)
                embeds.append(part.to(embed.weight.dtype))
                kinds.append(torch.full((len(part),), kind))
                frame_ids.append(torch.full((len(part),), frame))
            lengths.append(sum(len(part) for part, _ in parts))
        kind, frame = torch.cat(kinds).to(device), torch.cat(frame_ids).to(device)
        n = len(kind)
        meta = Tokens(kind=kind, frame=frame,
                      index=torch.arange(state.next_index, state.next_index + n, device=device),
                      keep=(kind == FRAME) & (frame == 0),                # 首帧画面永久保留
                      key=(kind == FRAME) & torch.isin(frame, torch.tensor(list(keyframes), device=device).long()))
        state.next_index += n
        hidden = forward_chunk(self.language, torch.cat(embeds)[None], torch.cat(positions, 1)[:, None].to(device),
                               meta, state, self.cfg.window, self.cfg.isolate_gdn)
        return hidden[0].split(lengths)

    def start(self, instruction, steps, demo):
        """前缀：任务指令 ·（成功画面）· 第 j 项文字 · 第 j 项示范关键帧 · [STEP]_j ……

        demo: [{"step": j 或 None, "views": [RGB 数组]}]，step 为 None 的是成功画面，附在指令后。
        """
        state = StreamState()
        items = [(PREFIX, f"Task: {instruction}\n")]
        success = [item for item in demo if item["step"] is None]
        if success:
            items.append((PREFIX, "Success looks like:"))
            items += [(PREFIX, image) for item in success for image in self.encode_images(item["views"])]
        for j, text in enumerate(steps):
            items.append((PREFIX, f"\nStep {j + 1}: {text}\n"))
            items += [(PREFIX, image) for item in demo if item["step"] == j
                      for image in self.encode_images(item["views"])]
            items.append((PREFIX, STEP_TOKEN))
        hidden = self._forward(state, items)
        state.steps = torch.cat([h for (_, content), h in zip(items, hidden, strict=True)
                                 if isinstance(content, int) and content == STEP_TOKEN])
        state.count_probs = state.steps.new_zeros(len(steps), MAX_COUNT, dtype=torch.float)   # 第一步回传全 0
        return state

    def _count_inputs(self, steps, probs):
        """[CNT] 的输入：e_CNT + Σ_j MLP([h_STEP_j; 上一步的 P(n_j ≥ 1..R)])。probs: [k, J, R] → [k, d]。"""
        return self.special[CNT_TOKEN] + self.count_feedback(
            torch.cat([steps.expand(len(probs), -1, -1), probs], -1)).sum(1)

    def _heads(self, steps, stage, count, progress):
        """一步的三个 readout hidden [d] → 各头的 logits。

        stage: [J+1]；count: [J, R]，第 (j, r) 个是"第 j 项已经做了至少 r 次"的 logit；progress: [bins]。
        """
        stage, count, progress = stage.float(), count.float(), progress.float()
        stage_logits = torch.cat([self.stage_head(torch.cat([stage.expand_as(steps), steps], -1)).squeeze(-1),
                                  self.none_head(stage)])
        scores = self.count_head(torch.cat([count.expand_as(steps), steps], -1))           # [J, 1]
        thresholds = torch.cat([self.count_first, self.count_first + F.softplus(self.count_gaps).cumsum(0)])
        return {"stage": stage_logits, "count": scores - thresholds, "progress": self.progress_head(progress)}

    def step(self, state, views):
        """追加一帧（多视角）和 [STG]·[CNT]·[P]，[CNT] 回传自己上一步的计数概率，返回各头的 logits。"""
        state.frame += 1
        steps = state.steps.float()
        items = [(FRAME, image) for image in self.encode_images(views)]
        items += [(STG, STG_TOKEN), (CNT, self._count_inputs(steps, state.count_probs[None])), (P, P_TOKEN)]
        out = self._heads(steps, *(h[0] for h in self._forward(state, items)[-3:]))
        state.count_probs = out["count"].detach().sigmoid()   # 截断：回传只当输入，不沿它反传
        return out

    def teacher_steps(self, state, sample, start, end, keyframes):
        """第 start…end−1 帧一次前向（stage 1）：[CNT] 回传上一帧的真值计数，所以各步的输入事先已知，
        可以像 LLM 的 teacher forcing 一样整段并行；窗口、可见性靠 mask，结果与逐步计算相同。"""
        steps = state.steps.float()
        truth = sample["count"].to(steps.device)
        previous = torch.cat([torch.zeros_like(truth[:1]), truth[:-1]])[start:end]
        counts = self._count_inputs(steps, previous)
        items, frames = [], []
        for i, t in enumerate(range(start, end)):
            step = [(FRAME, image) for image in self.encode_images(sample["frames"][t])]
            step += [(STG, STG_TOKEN), (CNT, counts[i : i + 1]), (P, P_TOKEN)]
            items += step
            frames += [t] * len(step)
        hidden = self._forward(state, items, frames, keyframes)
        state.frame = end - 1
        readouts = [h for (kind, _), h in zip(items, hidden, strict=True) if kind in (STG, CNT, P)]
        return [self._heads(steps, *(h[0] for h in readouts[i : i + 3])) for i in range(0, len(readouts), 3)]

    def rollout(self, sample, grad_from=0, keyframes=None, teacher=False):
        """跑完 sample 的前缀和全部帧；grad_from 之前的帧不求梯度（热身），返回之后各帧的输出。

        每步之后标关键帧（默认 cfg.keyframes）：label 用样本给的标注（子任务边界，训练时另有随机加入的帧），
        model 用 [CNT] 任一概率变化超过 δ。
        teacher（stage 1）：[CNT] 回传真值计数，热身按窗口长度分段、带梯度的一段整段并行，只用标注关键帧。
        """
        cfg = self.cfg
        keyframes = keyframes or cfg.keyframes
        if teacher and keyframes != "label":
            raise ValueError("teacher forcing 只支持标注关键帧")
        state = self.start(sample["instruction"], sample["steps"], sample["demo"])
        if teacher:
            real = set(sample["keyframes"].nonzero().flatten().tolist())
            extra = set(sample["extra_keyframes"].nonzero().flatten().tolist())
            # 仅在实际淘汰发生后切段，避免用未来的关键帧数量提前删除过去的证据。
            boundaries = {0, grad_from, len(sample["frames"]), *range(0, grad_from, cfg.window)}
            seen_real, active_extra = 0, []
            for t in sorted(real | extra):
                seen_real += t in real
                if t in extra:
                    active_extra.append(t)
                overflow = seen_real + len(active_extra) - cfg.max_keyframes
                if overflow > 0 and active_extra:
                    active_extra = active_extra[overflow:]
                    boundaries.add(t + 1)
            boundaries = sorted(boundaries)
            outputs = []
            for start, end in pairwise(boundaries):
                with torch.set_grad_enabled(torch.is_grad_enabled() and start >= grad_from):
                    out = self.teacher_steps(state, sample, start, end, real | extra)
                if start >= grad_from:
                    outputs += out
                state.keyframes.extend(sorted(t for t in real if start <= t < end))
                state.extra_keyframes.extend(sorted(t for t in extra if start <= t < end))
                state.evict(cfg.window, cfg.max_keyframes)
            return outputs
        outputs = []
        for t, views in enumerate(sample["frames"]):
            previous = state.count_probs
            with torch.set_grad_enabled(torch.is_grad_enabled() and t >= grad_from):
                out = self.step(state, views)
            if t >= grad_from:
                outputs.append(out)
            counts = state.count_probs
            real = (keyframes == "label" and sample["keyframes"][t]) or (
                keyframes == "model" and t > 0 and (counts - previous).abs().max() > cfg.delta)
            extra = keyframes == "label" and sample["extra_keyframes"][t]
            if real or extra:
                state.meta.key |= (state.meta.frame == t) & (state.meta.kind == FRAME)
                (state.keyframes if real else state.extra_keyframes).append(t)
            state.evict(cfg.window, cfg.max_keyframes)
        return outputs
