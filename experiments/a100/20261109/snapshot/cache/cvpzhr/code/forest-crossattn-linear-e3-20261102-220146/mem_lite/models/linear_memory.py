"""Qwen GatedDeltaNet 分块续接；只保存固定大小的卷积和递归状态。

沿用远程 ProgressLite/V10 的 initial_state 与卷积前缀续接方式。
不引入其 novelty gate、只读 [P] 或 Full Attention 窗口。
"""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F


@dataclass
class LinearMemory:
    # layer -> (最近 kernel_size-1 个有效 token 的投影, GatedDeltaNet 状态)
    layers: dict = field(default_factory=dict)

    def detach(self):
        return LinearMemory({i: tuple(t.detach() for t in pair) for i, pair in self.layers.items()})


def gated_delta_forward(layer, hidden, valid, conv=None, recurrent=None):
    """整段计算并返回新状态；padding 不进入卷积、不写入也不衰减递归状态。"""
    batch, length, _ = hidden.shape
    hidden = hidden.masked_fill(~valid[..., None], 0)
    mixed = layer.in_proj_qkv(hidden).transpose(1, 2)
    width = layer.conv_kernel_size - 1
    if conv is None:
        conv = mixed.new_zeros(batch, mixed.shape[1], width)

    # 将有效 token 紧凑排列，衔接上一帧卷积尾部；随后恢复当前帧的 padding 布局。
    order = (~valid).int().argsort(dim=-1, stable=True)
    packed = mixed.gather(2, order[:, None].expand_as(mixed))
    prefix = torch.cat((conv.to(mixed.dtype), packed), dim=-1)
    filtered = F.silu(F.conv1d(prefix, layer.conv1d.weight, layer.conv1d.bias, groups=mixed.shape[1]))
    offsets = valid.sum(-1, keepdim=True) + torch.arange(width, device=hidden.device)
    next_conv = prefix.gather(2, offsets[:, None].expand(-1, mixed.shape[1], -1))
    positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
    mixed = filtered.gather(2, positions[:, None].expand(-1, filtered.shape[1], -1))
    mixed = mixed.transpose(1, 2).masked_fill(~valid[..., None], 0)
    query, key, value = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, layer.head_k_dim)
    key = key.reshape(batch, length, -1, layer.head_k_dim)
    value = value.reshape(batch, length, -1, layer.head_v_dim)
    beta = layer.in_proj_b(hidden).sigmoid().masked_fill(~valid[..., None], 0)
    g = -layer.A_log.float().exp() * F.softplus(layer.in_proj_a(hidden).float() + layer.dt_bias)
    g = g.masked_fill(~valid[..., None], 0)
    groups = layer.num_v_heads // layer.num_k_heads
    query, key = query.repeat_interleave(groups, dim=2), key.repeat_interleave(groups, dim=2)
    # FLA 的 checkpoint 反传可能运行在另一个线程；allocator 是线程局部的。
    if hidden.is_cuda and layer.chunk_gated_delta_rule.__module__.startswith("fla."):
        import triton

        if hasattr(triton, "set_allocator"):
            triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.uint8,
                                                                           device=hidden.device))
    out, next_recurrent = layer.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=recurrent,
        output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    z = layer.in_proj_z(hidden).reshape(-1, layer.head_v_dim)
    out = layer.norm(out.reshape(-1, layer.head_v_dim), z).reshape(batch, length, -1)
    return layer.out_proj(out), next_conv, next_recurrent


def linear_decoder_forward(decoder, hidden, valid, conv, recurrent):
    mixed, conv, recurrent = gated_delta_forward(decoder.linear_attn, decoder.input_layernorm(hidden),
                                                valid, conv, recurrent)
    hidden = hidden + mixed
    hidden = hidden + decoder.mlp(decoder.post_attention_layernorm(hidden))
    return hidden, conv, recurrent
