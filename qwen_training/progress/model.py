# ruff: noqa: RUF002, RUF003
"""流式进度模型：Qwen3.5 底座 + [STEP]/[STG]/[CNT]/[P] readout（proposal_progress 3.2–3.5、第 4 节）。

- 视觉塔、词嵌入冻结；语言模型线性层 LoRA；GDN 的 in_proj_a/b、A_log、dt_bias 全量训练。
- 新 token 的 embedding、计数阈值和 MLP 头全量训练，与 LoRA 一起保存。
"""

import math

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

    def _forward(self, state, items):
        """items: [(kind, 内容)]，内容是文本、特殊 token 序号、embedding [1, d] 或 (图像特征, (h, w))。

        拼成一段接到缓存后前向，返回每个 item 对应的 hidden。图像前后加 vision_start、vision_end；
        位置按 Qwen3.5 的 get_rope_index：文本三轴相同，图像为 (起点, 起点 + 行, 起点 + 列)。
        """
        device, embed, config = self.special.device, self.language.embed_tokens, self.backbone.config
        marks = embed(torch.tensor([config.vision_start_token_id, config.vision_end_token_id], device=device))
        embeds, positions, kinds, lengths = [], [], [], []
        for kind, content in items:
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
            lengths.append(sum(len(part) for part, _ in parts))
        kind = torch.cat(kinds).to(device)
        n = len(kind)
        meta = Tokens(kind=kind, frame=torch.full((n,), state.frame, device=device),
                      index=torch.arange(state.next_index, state.next_index + n, device=device),
                      keep=(kind == FRAME) & (state.frame == 0),          # 首帧画面永久保留
                      key=torch.zeros(n, dtype=torch.bool, device=device))
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
        return state

    def step(self, state, views):
        """追加一帧（多视角）和 [STG]·[CNT]·[P]，返回各头的 logits。

        stage: [J+1]；count: [J, R]，第 (j, r) 个是"第 j 项已经做了至少 r 次"的 logit；progress: [bins]。
        """
        state.frame += 1
        items = [(FRAME, image) for image in self.encode_images(views)]
        steps = state.steps.float()
        if state.count_probs is None:
            state.count_probs = steps.new_zeros(len(steps), MAX_COUNT)
        count_input = self.special[CNT_TOKEN : CNT_TOKEN + 1] + self.count_feedback(
            torch.cat([steps, state.count_probs], -1)).sum(0, keepdim=True)
        items += [(STG, STG_TOKEN), (CNT, count_input), (P, P_TOKEN)]
        stage, count, progress = (h[0].float() for h in self._forward(state, items)[-3:])
        stage_logits = torch.cat([self.stage_head(torch.cat([stage.expand_as(steps), steps], -1)).squeeze(-1),
                                  self.none_head(stage)])
        scores = self.count_head(torch.cat([count.expand_as(steps), steps], -1))           # [J, 1]
        thresholds = torch.cat([self.count_first, self.count_first + F.softplus(self.count_gaps).cumsum(0)])
        count_logits = scores - thresholds
        state.count_probs = count_logits.detach().sigmoid()   # 截断：回传只当输入，不沿它反传
        return {"stage": stage_logits, "count": count_logits, "progress": self.progress_head(progress)}

    def rollout(self, sample, grad_from=0, keyframes=None):
        """跑完 sample 的前缀和全部帧；grad_from 之前的帧不求梯度（热身），返回之后各帧的输出。

        每步之后标关键帧（默认 cfg.keyframes）：label 用样本给的标注（子任务边界，训练时另有随机加入的帧），
        model 用 [CNT] 任一概率变化超过 δ。
        """
        cfg = self.cfg
        keyframes = keyframes or cfg.keyframes
        state = self.start(sample["instruction"], sample["steps"], sample["demo"])
        outputs = []
        for t, views in enumerate(sample["frames"]):
            previous = state.count_probs
            with torch.set_grad_enabled(torch.is_grad_enabled() and t >= grad_from):
                out = self.step(state, views)
            if t >= grad_from:
                outputs.append(out)
            counts = state.count_probs
            real = (keyframes == "label" and sample["keyframes"][t]) or (
                keyframes == "model" and previous is not None and (counts - previous).abs().max() > cfg.delta)
            extra = keyframes == "label" and sample["extra_keyframes"][t]
            if real or extra:
                state.meta.key |= (state.meta.frame == t) & (state.meta.kind == FRAME)
                (state.keyframes if real else state.extra_keyframes).append(t)
            state.evict(cfg.window, cfg.max_keyframes)
        return outputs
