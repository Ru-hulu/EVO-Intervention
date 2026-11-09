# ruff: noqa: RUF002, RUF003
"""流式进度模型：Qwen3.5 底座 + [STEP]/[STG]/[CNT]/[P] readout（proposal_progress 3.2–3.5、第 4 节）。

- 视觉塔、词嵌入冻结；语言模型线性层 LoRA；GDN 的 in_proj_a/b、A_log、dt_bias 全量训练。
- 新 token 的 embedding、计数阈值和 MLP 头全量训练，与 LoRA 一起保存。
"""

import math
from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from qwen_training.datasets.rmbench.labels import MAX_COUNT
from qwen_training.progress.stream import CNT, FRAME, PREFIX, STG, P, StreamState, Tokens, forward_chunk


def resize_images(images, width, height):
    """Downscale RGB images to a bounding box without changing aspect ratio or view count."""
    if width == height == 0:
        return images
    if width <= 0 or height <= 0:
        raise ValueError("image_max_width and image_max_height must both be positive or both zero")
    result = []
    for image in images:
        h, w = image.shape[:2]
        scale = min(1, width / w, height / h)
        if scale < 1:
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BOX)).copy()
        result.append(image)
    return result


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


def cumulative(counts):
    """次数 [..., J] → 累积编码 [..., J, R]：第 r 个为"至少 r 次"，与 level 模式回传的概率同形。"""
    return (counts[..., None] >= torch.arange(1, MAX_COUNT + 1, device=counts.device)).float()


class Standardize(nn.Module):
    """按特征维标准化：(x − 均值) / 标准差，截断到 ±clip；统计量训练时用滑动平均更新（不求梯度），验证时固定。

    readout hidden 经过最终 RMSNorm 的逐维缩放，少数维度数值远大于其余维度，会主导计数头的输入。
    统计量是跨调用的一、二阶矩 E[x]、E[x²]，方差 = E[x²] − E[x]²。不能用每次调用内的方差：一次调用的各行是
    同一步的 J 个项，前半 h_CNT 在各项间完全相同，调用内方差恒为 0，这些维度会被放大约 300 倍后截断，只剩符号。
    训练前用 fit 设定汇总的统计量（见 train.fit_count_norm）；fit 之前原样输出、不更新。
    momentum 按调用（一步）计，1e-3 约平均最近 1000 步，即十来条 episode。
    """

    def __init__(self, width, device, momentum=1e-3, clip=5.0):
        super().__init__()
        self.momentum, self.clip = momentum, clip
        self.register_buffer("mean", torch.zeros(width, device=device))
        self.register_buffer("square", torch.ones(width, device=device))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool, device=device))

    @property
    def var(self):
        return (self.square - self.mean**2).clamp(min=0)

    def fit(self, mean, square):
        self.mean.copy_(mean)
        self.square.copy_(square)
        self.initialized.fill_(True)

    def forward(self, x):
        if not self.initialized:
            return x
        if self.training:
            with torch.no_grad():
                flat = x.detach().reshape(-1, x.shape[-1])
                self.mean.lerp_(flat.mean(0), self.momentum)
                self.square.lerp_(flat.pow(2).mean(0), self.momentum)
        return ((x - self.mean) / (self.var + 1e-5).sqrt()).clamp(-self.clip, self.clip)


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
        self.event = cfg.count_mode == "event"
        if cfg.count_mode not in ("level", "event"):
            raise ValueError(f"count_mode 只能是 level 或 event, 而不是 {cfg.count_mode!r}")
        self.count_head, self.progress_head = head(2 * width, 3 if self.event else 1), head(width, cfg.progress_bins)
        self.count_norm = Standardize(2 * width, device) if cfg.count_norm else nn.Identity()
        # 累积计数：P(第 j 项做了至少 r 次) = σ(g_j − θ_r)，θ_1 < … < θ_R 由首项加 softplus 间隔保证递增，初始为 0…R−1。
        self.count_first = nn.Parameter(torch.zeros(1, device=device))
        self.count_gaps = nn.Parameter(torch.full((MAX_COUNT - 1,), math.log(math.e - 1), device=device))
        self.register_buffer("bins", torch.linspace(0, 1, cfg.progress_bins, device=device), persistent=False)

        for layer in language.layers:
            for parent in layer.modules():
                for name, child in list(parent.named_children()):
                    if name in LORA_TARGETS and isinstance(child, nn.Linear):
                        setattr(parent, name, LoRALinear(child, cfg.lora_rank, cfg.lora_alpha))
        self.gates = []   # (参数, 初始值)；gate_l2 > 0 时训练加 L2，让它们不远离初始值
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

    def trainable_state(self):
        """checkpoint 保存的部分：可训练参数，加上计数头输入标准化的统计量。"""
        names = {name for name, param in self.named_parameters() if param.requires_grad}
        names |= {f"count_norm.{name}" for name, _ in self.count_norm.named_buffers()}
        return {key: value.detach().cpu() for key, value in self.state_dict().items() if key in names}

    def load_trainable(self, weights):
        """载入 trainable_state() 保存的部分；结构变化时拒绝静默漏载。"""
        expected = set(self.trainable_state())
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
        images = resize_images(images, self.cfg.image_max_width, self.cfg.image_max_height)
        inputs = self.image_processor(images=images, return_tensors="pt")
        grids = inputs["image_grid_thw"].to(self.special.device)
        visual = self.backbone.model.visual
        features = self.backbone.model.get_image_features(
            inputs["pixel_values"].to(self.special.device, visual.dtype), grids).pooler_output
        merge = visual.spatial_merge_size
        return [(feature, (int(grid[1]) // merge, int(grid[2]) // merge))
                for feature, grid in zip(features, grids, strict=True)]

    def _write_strength(self, state, images):
        """一帧各视角的逐 patch 写入强度：和上一帧同视角比较 1−cos，过死区后线性升到 1；首帧全写（None）。"""
        if not self.cfg.novelty_gate:
            return [None] * len(images)
        low, high = self.cfg.novelty_low, self.cfg.novelty_high
        previous, state.last_images = state.last_images, [feature for feature, _ in images]
        if previous is None:
            return [None] * len(images)
        return [((1 - F.cosine_similarity(feature.float(), before.float(), -1) - low) / (high - low)).clamp(0, 1)
                for (feature, _), before in zip(images, previous, strict=True)]

    def _forward(self, state, items, frames=None, keyframes=(), strengths=None):
        """items: [(kind, 内容)]，内容是文本、特殊 token 序号、embedding [1, d] 或 (图像特征, (h, w))。

        拼成一段接到缓存后前向，返回每个 item 对应的 hidden。图像前后加 vision_start、vision_end；
        位置按 Qwen3.5 的 get_rope_index：文本三轴相同，图像为 (起点, 起点 + 行, 起点 + 列)。
        frames：每个 item 所在的帧号，默认都是 state.frame（一段多步并行时逐个给出）；keyframes 中的帧标为关键帧。
        strengths：每个 item 的写入强度（见 _write_strength），None 为全写；视觉起止符取该图 patch 的最大值。
        """
        device, embed, config = self.special.device, self.language.embed_tokens, self.backbone.config
        marks = embed(torch.tensor([config.vision_start_token_id, config.vision_end_token_id], device=device))
        frames = frames or [state.frame] * len(items)
        strengths = strengths or [None] * len(items)
        embeds, positions, kinds, frame_ids, lengths, gates = [], [], [], [], [], []
        for (kind, content), frame, strength in zip(items, frames, strengths, strict=True):
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
                gates.append(torch.ones(len(part), device=device) if strength is None
                             else strength if grid is not None else strength.max().expand(len(part)))
            lengths.append(sum(len(part) for part, _ in parts))
        kind, frame = torch.cat(kinds).to(device), torch.cat(frame_ids).to(device)
        n = len(kind)
        meta = Tokens(kind=kind, frame=frame,
                      index=torch.arange(state.next_index, state.next_index + n, device=device),
                      keep=(kind == FRAME) & (frame == 0),                # 首帧画面永久保留
                      key=(kind == FRAME) & torch.isin(frame, torch.tensor(list(keyframes), device=device).long()))
        state.next_index += n
        hidden = forward_chunk(self.language, torch.cat(embeds)[None], torch.cat(positions, 1)[:, None].to(device),
                               meta, state, self.cfg.window, self.cfg.isolate_gdn,
                               self.cfg.progress_reads_count, self.cfg.count_reads_history,
                               self.cfg.stage_reads_history, self.cfg.progress_reads_history,
                               self.cfg.activation_checkpointing,
                               torch.cat(gates) if self.cfg.novelty_gate else None)
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
        state.counts = torch.zeros(len(steps), dtype=torch.long, device=state.steps.device)  # event：累加的次数
        return state

    def _count_inputs(self, steps, probs):
        """[CNT] 的输入：e_CNT + Σ_j MLP([h_STEP_j; 上一步的 P(n_j ≥ 1..R)])。probs: [k, J, R] → [k, d]。"""
        if not self.cfg.count_feedback:
            probs = torch.zeros_like(probs)
        return self.special[CNT_TOKEN] + self.count_feedback(
            torch.cat([steps.expand(len(probs), -1, -1), probs], -1)).sum(1)

    def _heads(self, steps, stage, count, progress):
        """单步 [d] 或多步 [T, d] 的 readout → STG [..., J+1]、CNT [..., J, R/3]、P [..., bins]。"""
        stage, count, progress = stage.float(), count.float(), progress.float()
        steps = steps.expand(*stage.shape[:-1], *steps.shape)
        stage_logits = self.stage_head(torch.cat([stage.unsqueeze(-2).expand_as(steps), steps], -1)).squeeze(-1)
        stage_logits = torch.cat([stage_logits, self.none_head(stage)], -1)
        features = torch.cat([count.unsqueeze(-2).expand_as(steps), steps], -1)
        if features.ndim == 3 and isinstance(self.count_norm, Standardize) and self.count_norm.training:
            # 保留逐帧更新滑动统计量的语义；MLP 仍整段并行。
            features = torch.stack([self.count_norm(frame) for frame in features])
        else:
            features = self.count_norm(features)
        scores = self.count_head(features)
        if not self.event:
            thresholds = torch.cat([self.count_first, self.count_first + F.softplus(self.count_gaps).cumsum(0)])
            scores = scores - thresholds
        return {"stage": stage_logits, "count": scores, "progress": self.progress_head(progress)}

    def step(self, state, views, count_probs=None):
        """追加一帧（多视角）和 [STG]·[CNT]·[P]，计数反馈由 cfg.count_feedback 控制，返回各头 logits。"""
        state.frame += 1
        steps = state.steps.float()
        images = self.encode_images(views)
        items = [(FRAME, image) for image in images]
        probs = state.count_probs if count_probs is None else count_probs.to(steps.device)
        items += [(STG, STG_TOKEN), (CNT, self._count_inputs(steps, probs[None])), (P, P_TOKEN)]
        strengths = [*self._write_strength(state, images), None, None, None]
        out = self._heads(steps, *(h[0] for h in self._forward(state, items, strengths=strengths)[-3:]))
        if self.event:   # 次数在模型外累加：这一步的输入次数加上预测的变化
            out["fed"] = state.counts if count_probs is None else probs.sum(-1).long()
            state.counts = (out["fed"] + out["count"].argmax(-1) - 1).clamp(0, MAX_COUNT)
            state.count_probs = cumulative(state.counts)
        else:
            state.count_probs = out["count"].detach().sigmoid()   # 截断：回传只当输入，不沿它反传
        return out

    def teacher_steps(self, state, sample, start, end, keyframes):
        """第 start…end−1 帧并行编码图像、计算主干和输出头。

        teacher forcing 使用 sample["fed"]；无反馈模式回传置零。各步输入事先已知，窗口与因果性由 mask 保持。"""
        steps = state.steps.float()
        fed = sample["fed"][start:end].to(steps.device)
        counts = self._count_inputs(steps, cumulative(fed))
        views = [sample["frames"][t] for t in range(start, end)]
        images = iter(self.encode_images([image for frame in views for image in frame]))   # 整段的图像一次编码
        items, frames, strengths = [], [], []
        for i, t in enumerate(range(start, end)):
            frame_images = [next(images) for _ in views[i]]
            step = [(FRAME, image) for image in frame_images]
            step += [(STG, STG_TOKEN), (CNT, counts[i : i + 1]), (P, P_TOKEN)]
            items += step
            frames += [t] * len(step)
            strengths += [*self._write_strength(state, frame_images), None, None, None]
        hidden = self._forward(state, items, frames, keyframes, strengths)
        state.frame = end - 1
        readouts = [h for (kind, _), h in zip(items, hidden, strict=True) if kind in (STG, CNT, P)]
        readouts = torch.stack([h[0] for h in readouts]).reshape(end - start, 3, -1)
        logits = self._heads(steps, *readouts.unbind(1))
        outputs = [{name: values[i] for name, values in logits.items()} for i in range(end - start)]
        if self.event:
            for out, n in zip(outputs, fed, strict=True):
                out["fed"] = n
        return outputs

    def rollout_segments(self, sample):
        """Yield one segment at a time; caller backpropagates before advancing the generator."""
        if self.cfg.teacher or self.cfg.keyframes != "none" or self.cfg.count_feedback or self.event:
            raise ValueError("multi-segment training requires no-feedback level mode and keyframes=none")
        sample = {**sample, "fed": torch.zeros_like(sample["fed"])}
        state = self.start(sample["instruction"], sample["steps"], sample["demo"])
        cursor = 0
        for start, end in sample["segment_ranges"]:
            for lo, hi, chunk, grad in [(cursor, start, self.cfg.warmup_chunk_size or self.cfg.chunk_size, False),
                                         (start, end, self.cfg.train_chunk_size or self.cfg.chunk_size, True)]:
                outputs = []
                with torch.set_grad_enabled(grad):
                    for a in range(lo, hi, chunk):
                        out = self.teacher_steps(state, sample, a, min(a + chunk, hi), set())
                        if grad:
                            outputs.extend(out)
                        state.evict(self.cfg.window, self.cfg.max_keyframes)
            yield outputs, start, end
            state.detach()
            cursor = end

    def rollout(self, sample, grad_from=0, keyframes=None, teacher=False):
        """跑完 sample 的前缀和全部帧；grad_from 之前的帧不求梯度（热身），返回之后各帧的输出。

        每步之后标关键帧（默认 cfg.keyframes）：label 用样本给的标注（子任务边界，训练时另有随机加入的帧），
        model 用 [CNT] 任一概率变化超过 δ。
        teacher（stage 1）：[CNT] 回传真值计数，按 chunk_size 分段并行，只用标注关键帧。
        keyframes=none、count_feedback=False、level 模式时也走分段并行（见 teacher_steps）。
        warmup_chunk_size / train_chunk_size 分别覆盖热身 / 反传的 chunk_size；0 表示沿用。
        不求梯度时全程用 warmup_chunk_size。
        训练 chunk 之间不 detach，segment 内的梯度保持连续。
        """
        cfg = self.cfg
        if cfg.warmup_chunk_size < 0 or cfg.train_chunk_size < 0:
            raise ValueError("warmup_chunk_size / train_chunk_size must be >= 0")
        warmup_chunk = cfg.warmup_chunk_size or cfg.chunk_size
        # 不求梯度时（评估）grad_from 之后也只是无梯度前向，按热身的 chunk 切
        train_chunk = (cfg.train_chunk_size or cfg.chunk_size) if torch.is_grad_enabled() else warmup_chunk
        if cfg.chunk_size < 1:
            raise ValueError("chunk_size 必须 > 0")
        if cfg.oracle_count_feedback and not cfg.count_feedback:
            raise ValueError("真值回传不能与关闭计数回传同时使用")
        keyframes = keyframes or cfg.keyframes
        if teacher and keyframes != "label":
            raise ValueError("teacher forcing 只支持标注关键帧")
        state = self.start(sample["instruction"], sample["steps"], sample["demo"])
        # 关闭计数回传（回传置零）且不留关键帧时，各步输入同样事先已知，按 chunk_size 分段并行，结果与逐步相同
        parallel = not teacher and keyframes == "none" and not cfg.count_feedback and not self.event
        if teacher or parallel:
            if parallel:
                sample = {**sample, "fed": torch.zeros(len(sample["frames"]), len(sample["steps"]), dtype=torch.long)}
                real, extra = set(), set()
            else:
                real = set(sample["keyframes"].nonzero().flatten().tolist())
                extra = set(sample["extra_keyframes"].nonzero().flatten().tolist())
            # 仅在实际淘汰发生后切段，避免用未来的关键帧数量提前删除过去的证据。
            # 热身从 0 对齐，反传从 grad_from 重新对齐；不改变窗口或截断段内梯度。
            boundaries = {0, grad_from, len(sample["frames"]),
                          *range(0, grad_from, warmup_chunk),
                          *range(grad_from, len(sample["frames"]), train_chunk)}
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
                truth = None
                if cfg.oracle_count_feedback:
                    truth = sample["count"][t - 1] if t else torch.zeros_like(sample["count"][0])
                out = self.step(state, views, count_probs=truth)
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
