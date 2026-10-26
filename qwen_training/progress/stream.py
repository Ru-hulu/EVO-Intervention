# ruff: noqa: RUF002, RUF003
"""流式前向：前缀一次算完，之后追加帧和 readout（proposal_progress 3.3、3.4）。

直接复用 Qwen3.5 各层的权重，只替换 token 混合部分：
- full attention 层：缓存只保留"前缀 + 首帧 + 关键帧 + 最近 N 帧"，按 token 类型给 mask；
- GDN 层：readout 位置 β=0、g=0（α=1），不写入、不衰减，也不进入短卷积窗口。
推理和 stage 2 每次追加一步；stage 1（teacher forcing）一次追加多步，mask 按每个 token 所在的帧计算，结果相同。
"""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

PREFIX, FRAME, STG, CNT, P = range(5)


@dataclass
class Tokens:
    """一段 token 的元信息，各字段都是长度相同的一维张量。"""

    kind: torch.Tensor   # PREFIX / FRAME / STG / CNT / P
    frame: torch.Tensor  # 所在帧号，前缀为 -1
    index: torch.Tensor  # 全局顺序，用于因果
    keep: torch.Tensor   # 首帧画面，永久保留
    key: torch.Tensor    # 关键帧画面

    def cat(self, other):
        return Tokens(*(torch.cat([getattr(self, f.name), getattr(other, f.name)]) for f in fields(self)))

    def select(self, mask):
        return Tokens(*(getattr(self, f.name)[mask] for f in fields(self)))


def visibility(query, keys, window):
    """[Lq, Lk] 布尔 mask，True 表示可见。规则见 proposal 3.4 的表。"""
    qk, kk = query.kind[:, None], keys.kind[None]
    reads_state = (qk == CNT) | (qk == P)
    in_window = keys.frame[None] > query.frame[:, None] - window
    frame = (kk == FRAME) & (in_window | keys.keep[None] | (keys.key[None] & reads_state))
    readout = in_window & (((kk == CNT) & reads_state) | (((kk == STG) | (kk == P)) & (qk == P)))
    itself = keys.index[None] == query.index[:, None]
    causal = keys.index[None] <= query.index[:, None]
    return causal & ((kk == PREFIX) | frame | readout | itself)


class StreamState:
    """一条 episode 的缓存。

    前缀的 K、V 单独存放、从不淘汰，训练时保留梯度；视频流部分的 K、V 按窗口和关键帧淘汰。
    GDN 层只有短卷积状态和递归状态。
    """

    def __init__(self):
        self.prefix, self.stream = {}, {}        # layer -> (K, V)，形状 [B, H, L, D]
        self.prefix_meta, self.meta = None, None
        self.conv, self.recurrent = {}, {}
        self.next_index = 0
        self.next_position = 0
        self.frame = -1
        self.keyframes = []                      # 真正的关键帧（标注边界或模型触发），不设上限
        self.extra_keyframes = []                # 训练时随机加入的帧
        self.steps = None                        # [J, d]，各 [STEP] 的 hidden
        self.count_probs = None                  # [J, R]，上一步的计数概率，回传给下一步的 [CNT]

    def evict(self, window, max_keyframes):
        """为下一帧淘汰：保留首帧、全部关键帧和最近 window-1 帧。

        真正的关键帧不设上限；关键帧总数超过 max_keyframes 时，从最早的随机加入帧删起。
        """
        while self.extra_keyframes and len(self.keyframes) + len(self.extra_keyframes) > max_keyframes:
            dropped = self.extra_keyframes.pop(0)
            self.meta.key &= self.meta.frame != dropped
        meta = self.meta
        keep = meta.keep | meta.key | (meta.frame > self.frame + 1 - window)
        if keep.all():
            return
        self.meta = meta.select(keep)
        for layer, (key, value) in self.stream.items():
            self.stream[layer] = (key[:, :, keep], value[:, :, keep])


def gated_delta_net(layer, hidden, write, state, layer_idx):
    """Qwen3_5GatedDeltaNet 的分段版本；write 为 False 的 token 只读不写。"""
    batch, length, _ = hidden.shape
    mixed = layer.in_proj_qkv(hidden).transpose(1, 2)            # [B, C, L]
    weight = layer.conv1d.weight.squeeze(1)                     # [C, k]
    width = weight.shape[-1] - 1
    history = state.conv.get(layer_idx)
    if history is None:
        history = mixed.new_zeros(batch, mixed.shape[1], width)
    # 每个 token 的卷积窗口 = 它之前最近 width 个写入 token + 它自己。
    history = torch.cat([history, mixed[:, :, write]], dim=-1)
    windows = history.unfold(-1, width, 1)                      # [B, C, 写入数 + 1, width]
    before = torch.cumsum(write, 0) - write.long()
    context = windows[:, :, before]                              # [B, C, L, width]
    # autocast 下 sum 输出 fp32，fla kernel 要求 bf16。
    mixed = F.silu((context * weight[:, None, :width]).sum(-1) + weight[:, -1:] * mixed).to(hidden.dtype)
    state.conv[layer_idx] = history[:, :, -width:]

    query, key, value = torch.split(mixed.transpose(1, 2), [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, layer.head_k_dim)
    key = key.reshape(batch, length, -1, layer.head_k_dim)
    value = value.reshape(batch, length, -1, layer.head_v_dim)
    beta = layer.in_proj_b(hidden).sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(layer.in_proj_a(hidden).float() + layer.dt_bias)
    reads = ~write[None, :, None]
    beta = beta.masked_fill(reads, 0)
    g = g.masked_fill(reads, 0)
    if layer.num_v_heads // layer.num_k_heads > 1:
        query = query.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
        key = key.repeat_interleave(layer.num_v_heads // layer.num_k_heads, dim=2)
    out, state.recurrent[layer_idx] = layer.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, initial_state=state.recurrent.get(layer_idx),
        output_final_state=True, use_qk_l2norm_in_kernel=True)
    z = layer.in_proj_z(hidden).reshape(-1, layer.head_v_dim)
    out = layer.norm(out.reshape(-1, layer.head_v_dim), z).reshape(batch, length, -1)
    return layer.out_proj(out)


def full_attention(attn, hidden, cos, sin, mask, state, layer_idx):
    """Qwen3_5Attention：K、V 接在前缀和视频流缓存之后，按 mask 计算。"""
    batch, length, _ = hidden.shape
    shape = (batch, length, -1, attn.head_dim)
    query, gate = torch.chunk(attn.q_proj(hidden).view(batch, length, -1, attn.head_dim * 2), 2, dim=-1)
    query = attn.q_norm(query).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
    value = attn.v_proj(hidden).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if state.prefix_meta is None:
        state.prefix[layer_idx] = (key, value)
    else:
        if layer_idx in state.stream:
            key = torch.cat([state.stream[layer_idx][0], key], dim=2)
            value = torch.cat([state.stream[layer_idx][1], value], dim=2)
        state.stream[layer_idx] = (key, value)
        key = torch.cat([state.prefix[layer_idx][0], key], dim=2)
        value = torch.cat([state.prefix[layer_idx][1], value], dim=2)
    # K、V 先展开到 query 的 head 数：带自定义 mask 时 GQA 只能走 math 后端，会为反传存下整个注意力矩阵
    # （2B 每层每步约 150 MB）；展开后可走 memory-efficient kernel。
    groups = attn.num_key_value_groups
    out = F.scaled_dot_product_attention(query, key.repeat_interleave(groups, 1), value.repeat_interleave(groups, 1),
                                         attn_mask=mask, scale=attn.scaling)
    out = out.transpose(1, 2).reshape(batch, length, -1) * torch.sigmoid(gate.reshape(batch, length, -1))
    return attn.o_proj(out)


def forward_chunk(language_model, embeds, positions, meta, state, window, isolate_gdn=True):
    """把一段 token 接到缓存后面，返回最后一层（已 norm）的 hidden。

    state 里还没有前缀时，这一段就作为前缀。positions: [3, B, L]。
    """
    cos, sin = language_model.rotary_emb(embeds, positions)
    stream = meta if state.meta is None else state.meta.cat(meta)
    keys = meta if state.prefix_meta is None else state.prefix_meta.cat(stream)
    mask = visibility(meta, keys, window)
    write = meta.kind <= FRAME if isolate_gdn else torch.ones_like(meta.kind, dtype=torch.bool)
    hidden = embeds
    for layer_idx, layer in enumerate(language_model.layers):
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        if layer.layer_type == "linear_attention":
            hidden = gated_delta_net(layer.linear_attn, hidden, write, state, layer_idx)
        else:
            hidden = full_attention(layer.self_attn, hidden, cos, sin, mask, state, layer_idx)
        hidden = residual + hidden
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    if state.prefix_meta is None:
        state.prefix_meta = meta
    else:
        state.meta = stream
    return language_model.norm(hidden)
