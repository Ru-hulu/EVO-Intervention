# ruff: noqa: RUF002, RUF003
"""只预测 p 的流式模型（proposal_progress_v10）：Qwen3.5 底座 + 一个 [P] readout。

前缀只有任务指令。每步追加当前帧各视角和 [P]；[P] 从 S、首帧和窗口读历史，不写入 S。
视觉塔、词嵌入冻结；语言模型线性层 LoRA；GDN 的门全量训练；[P] 和 MLP 头全量训练。
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from qwen_training.progress_lite.stream import FRAME, PREFIX, P, StreamState, Tokens, forward_chunk

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                "gate_proj", "up_proj", "down_proj")
GATE_PARAMS = ("in_proj_a", "in_proj_b", "A_log", "dt_bias")
# 只缩小不放大：处理器默认把面积小于 256² 的图放大，AgiBot 的小图会被放回去
IMAGE_AREA = {"shortest_edge": 32 * 32, "longest_edge": 4096 * 4096}


def autocast(cfg):
    """关掉类型转换缓存：验证时缓存的 bf16 副本没有梯度，之后的带梯度步会拿不到梯度。"""
    device = torch.device(cfg.device).type
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda", cache_enabled=False)


def resize_images(images, width, height):
    """按比例缩到不超过 width×height，视角数不变；宽高都为 0 时不缩放。"""
    if width == height == 0:
        return images
    result = []
    for image in images:
        h, w = image.shape[:2]
        scale = min(1, width / w, height / h)
        if scale < 1:
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            image = np.asarray(Image.fromarray(image).resize(size, Image.Resampling.BOX)).copy()
        result.append(image)
    return result


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


class ProgressLite(nn.Module):
    def __init__(self, backbone, image_processor, tokenizer, cfg):
        super().__init__()
        self.backbone, self.image_processor, self.tokenizer, self.cfg = backbone, image_processor, tokenizer, cfg
        backbone.requires_grad_(False)
        language = backbone.model.language_model
        width, device = backbone.config.text_config.hidden_size, language.embed_tokens.weight.device
        scale = language.embed_tokens.weight.float().std().item()
        self.token = nn.Parameter(torch.randn(1, width, device=device) * scale)
        self.head = nn.Sequential(nn.Linear(width, cfg.head_hidden), nn.GELU(),
                                  nn.Linear(cfg.head_hidden, cfg.progress_bins)).to(device)
        self.register_buffer("bins", torch.linspace(0, 1, cfg.progress_bins, device=device), persistent=False)
        for layer in language.layers:
            for parent in layer.modules():
                for name, child in list(parent.named_children()):
                    if name in LORA_TARGETS and isinstance(child, nn.Linear):
                        setattr(parent, name, LoRALinear(child, cfg.lora_rank, cfg.lora_alpha))
        self.gates = []
        for name, param in language.named_parameters():
            if any(part in GATE_PARAMS for part in name.split(".")):
                param.data = param.data.float()
                param.requires_grad_(True)
                self.gates.append(param)

    @classmethod
    def from_pretrained(cls, path, cfg, device="cuda"):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device)
        processor = AutoProcessor.from_pretrained(path)
        return cls(backbone, processor.image_processor, processor.tokenizer, cfg)

    def trainable_state(self):
        names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {key: value.detach().cpu() for key, value in self.state_dict().items() if key in names}

    def load_trainable(self, weights):
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
        device, visual = self.token.device, self.backbone.model.visual
        inputs = self.image_processor(images=resize_images(images, self.cfg.image_max_width, self.cfg.image_max_height),
                                      size=IMAGE_AREA, return_tensors="pt")
        grids = inputs["image_grid_thw"].to(device)
        features = self.backbone.model.get_image_features(inputs["pixel_values"].to(device, visual.dtype),
                                                          grids).pooler_output
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

    def _forward(self, state, parts):
        """parts: [(kind, 帧号, embedding [n, d], 图像网格或 None, 写入强度 [n] 或 None)]。

        拼成一段接到缓存后前向，返回 [P] 的最后一层 hidden。位置按 Qwen3.5 的 get_rope_index：
        文本三轴相同，图像为 (起点, 起点 + 行, 起点 + 列)。
        """
        device, dtype = self.token.device, self.language.embed_tokens.weight.dtype
        embeds, positions, kinds, frames, gates = [], [], [], [], []
        for kind, frame, embed, grid, strength in parts:
            start, length = state.next_position, len(embed)
            if grid is None:
                positions.append(torch.arange(start, start + length).expand(3, -1))
                state.next_position += length
            else:
                height, width = grid
                positions.append(torch.stack([torch.full((length,), start),
                                              torch.arange(start, start + height).repeat_interleave(width),
                                              torch.arange(start, start + width).repeat(height)]))
                state.next_position += max(height, width)
            embeds.append(embed.to(dtype))
            kinds.append(torch.full((length,), kind))
            frames.append(torch.full((length,), frame))
            gates.append(torch.ones(length, device=device) if strength is None else strength)
        kind, frame = torch.cat(kinds).to(device), torch.cat(frames).to(device)
        n = len(kind)
        meta = Tokens(kind=kind, frame=frame, index=torch.arange(state.next_index, state.next_index + n, device=device),
                      keep=(kind == FRAME) & (frame == 0))               # 首帧画面永久保留
        state.next_index += n
        hidden = forward_chunk(self.language, torch.cat(embeds)[None], torch.cat(positions, 1)[:, None].to(device),
                               meta, state, self.cfg.window,
                               activation_checkpointing=self.cfg.activation_checkpointing,
                               strength=torch.cat(gates) if self.cfg.novelty_gate else None)
        return hidden[0][kind == P]

    def start(self, instruction):
        """前缀只有任务指令，返回这条轨迹的缓存。"""
        ids = self.tokenizer(f"Task: {instruction}\n", add_special_tokens=False)["input_ids"]
        state = StreamState()
        self._forward(state, [(PREFIX, -1, self.language.embed_tokens(torch.tensor(ids, device=self.token.device)),
                               None, None)])
        return state

    def steps(self, state, frames, start):
        """第 start 帧起的一段帧并行前向，frames: [[各视角 RGB 数组]]，返回各帧 p 的 logits [T, bins]。"""
        embed, config = self.language.embed_tokens, self.backbone.config
        marks = embed(torch.tensor([config.vision_start_token_id, config.vision_end_token_id], device=self.token.device))
        images = iter(self.encode_images([image for views in frames for image in views]))   # 整段的图像一次编码
        parts = []
        for t, views in enumerate(frames, start):
            frame_images = [next(images) for _ in views]
            for (feature, grid), strength in zip(frame_images, self._write_strength(state, frame_images), strict=True):
                edge = None if strength is None else strength.mean()[None]   # 视觉起止符：写入量和整张图的变化成比例
                parts += [(FRAME, t, marks[:1], None, edge), (FRAME, t, feature, grid, strength),
                          (FRAME, t, marks[1:], None, edge)]
            parts.append((P, t, self.token, None, None))
        return self.head(self._forward(state, parts).float())

    def advance(self, state, frames, begin, end, chunk):
        """把第 begin…end−1 帧按 chunk 接到缓存后面，每段之后淘汰窗口外的帧。返回各段 logits 的列表。"""
        outputs = []
        for start in range(begin, end, chunk):
            stop = min(start + chunk, end)
            outputs.append(self.steps(state, frames[start:stop], start))
            state.frame = stop - 1
            state.evict(self.cfg.window)
        return outputs

    def rollout(self, sample):
        """跑完整条样本，返回 p 的 logits [T, bins]。"""
        state = self.start(sample["instruction"])
        return torch.cat(self.advance(state, sample["frames"], 0, len(sample["frames"]), self.cfg.eval_chunk_size))

    def rollout_segments(self, sample):
        """逐段产出 (logits, 起点, 终点)，调用方先反传再推进；段之后切断全部缓存的梯度，S 的数值继续往后传。"""
        state = self.start(sample["instruction"])
        for start, end in sample["segment_ranges"]:
            yield torch.cat(self.advance(state, sample["frames"], start, end, self.cfg.train_chunk_size)), start, end
            state.detach()

    def expected(self, logits):
        """分箱分布的期望，即预测的 p。"""
        return (F.softmax(logits.float(), -1) * self.bins).sum(-1)

    def loss(self, logits, target):
        """p 的标签摊到相邻两个分箱上（线性插值），对分箱分布做交叉熵。"""
        bins = len(self.bins)
        position = target.clamp(0, 1) * (bins - 1)
        low = position.floor().long().clamp(max=bins - 2)
        high = (position - low)[:, None]
        soft = torch.zeros(len(low), bins, device=low.device)
        soft.scatter_(1, low[:, None], 1 - high).scatter_add_(1, low[:, None] + 1, high)
        return -(soft * F.log_softmax(logits.float(), -1)).sum(-1).mean()
