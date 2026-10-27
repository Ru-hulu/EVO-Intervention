"""共享的冻结 Qwen3.5 视觉编码器与 StreamForest 时空位置适配。"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def _vision_state(model_path):
    from safetensors import safe_open

    model_path = Path(model_path)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted({name for key, name in weight_map.items() if key.startswith("model.visual.")})
    else:
        shards = ["model.safetensors"]
    state = {}
    for shard in shards:
        with safe_open(model_path / shard, framework="pt", device="cpu") as source:
            for key in sorted(source.keys()):
                if key.startswith("model.visual."):
                    tensor = source.get_tensor(key)
                    state[key.removeprefix("model.visual.")] = tensor
    return state


def _vision_sha256(state):
    # 对实际使用的视觉权重取摘要，不重复读取整个语言模型分片。
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_vision_source(model_path, history_source):
    if _vision_sha256(_vision_state(model_path)) != history_source["vision_sha256"]:
        raise ValueError("历史缓存的视觉权重与 Qwen 基座不一致，请使用原基座或重新构造缓存")


def load_vision_encoder(model_path, device):
    """只读取Qwen3.5原始检查点的ViT/merger权重，不加载语言模型或LoRA。"""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = AutoConfig.from_pretrained(model_path, local_files_only=True).vision_config
    # 只延迟分配参数；非持久RoPE缓冲必须按配置真实初始化，不能留在meta设备。
    with init_empty_weights(include_buffers=False):
        model = Qwen3_5VisionModel._from_config(config, attn_implementation="sdpa")
    state = _vision_state(model_path)
    vision_hash = _vision_sha256(state)
    model.load_state_dict(state, assign=True)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).requires_grad_(False).eval()
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    return model, processor, vision_hash


def sinusoidal_positions(indices, width):
    frequencies = 10000 ** (-torch.arange(0, width, 2, device=indices.device, dtype=torch.float32) / width)
    angles = indices.float()[..., None] * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :width]


def spatiotemporal_positions(height, width, frame_position, hidden_size, device):
    """沿用 StreamForest 的时间/空间分区，支持矩形网格和不循环的帧位置。"""
    temporal_dim = hidden_size // 4
    row_dim = (hidden_size - temporal_dim) // 2
    col_dim = hidden_size - temporal_dim - row_dim
    rows, cols = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    time = torch.as_tensor(frame_position, device=device).expand(height * width)
    return torch.cat((sinusoidal_positions(time, temporal_dim),
                      sinusoidal_positions(rows.flatten(), row_dim),
                      sinusoidal_positions(cols.flatten(), col_dim)), dim=-1)


class QwenFrameEncoder(nn.Module):
    """只持有原生 ViT/merger，可与已有 VLM 共享；不创建语言模型或 LoRA。"""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual.requires_grad_(False).eval()
        self.eval()

    @property
    def hidden_size(self):
        return self.visual.config.out_hidden_size

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, pixel_values, image_grid_thw, frame_positions):
        """processor 的逐帧图像输入 → 带位置的 [N_i, D] 列表。

        frame_positions 由调用方传入绝对帧位置；分批编码时不能从零重计。
        每个 grid 的 T=1，N_i 为原生视觉 merger 之后的空间 token 数。
        """
        self.visual.eval()
        parameter = next(self.visual.parameters())
        grid = image_grid_thw.to(parameter.device)
        pixels = pixel_values.to(device=parameter.device, dtype=parameter.dtype)
        # 固定按视觉权重的 dtype 编码，外层训练/推理的 autocast 不得改变缓存。
        with torch.autocast(parameter.device.type, enabled=False):
            tokens = self.visual(pixels, grid_thw=grid, return_dict=True).pooler_output
        merge = self.visual.config.spatial_merge_size
        counts = (grid.prod(-1) // merge**2).tolist()
        frames = []
        for features, (_, height, width), position in zip(
            tokens.split(counts), grid.tolist(), frame_positions, strict=True,
        ):
            positions = spatiotemporal_positions(height // merge, width // merge, position,
                                                self.hidden_size, parameter.device)
            frames.append(features + positions.to(features.dtype))
        return frames
