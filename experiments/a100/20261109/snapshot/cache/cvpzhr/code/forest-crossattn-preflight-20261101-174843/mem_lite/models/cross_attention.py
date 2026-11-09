"""Gated reads of frozen Forest memory between native Qwen decoder layers."""

import torch
from torch import nn
from torch.nn import functional as F


class GatedHistoryAttention(nn.Module):
    def __init__(self, hidden_size, history_dim, attention_dim=512, num_heads=8):
        super().__init__()
        if attention_dim < 1 or num_heads < 1 or attention_dim % num_heads:
            raise ValueError("cross_attention_dim 必须是 cross_attention_heads 的正整数倍")
        self.num_heads = num_heads
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(history_dim)
        self.q_proj = nn.Linear(hidden_size, attention_dim, bias=False)
        self.k_proj = nn.Linear(history_dim, attention_dim, bias=False)
        self.v_proj = nn.Linear(history_dim, attention_dim, bias=False)
        self.out_proj = nn.Linear(attention_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("frequencies", 10000 ** (-torch.arange(0, history_dim, 2).float() / history_dim),
                             persistent=False)

    def forward(self, hidden, memory, valid, query_valid):
        active = valid.any(-1)
        if memory.shape[1] == 0 or not active.any():
            return hidden
        # Padding may contain arbitrary values, including NaN; sanitize before projection.
        memory = memory.detach().masked_fill(~valid[..., None], 0).to(self.k_proj.weight.dtype)
        memory = self.memory_norm(memory)
        # Cache order is long -> short -> recent. These are ordinal positions, not timestamps.
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0)
        angles = positions.float()[..., None] * self.frequencies.float()
        encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :memory.shape[-1]]
        query = self.q_proj(self.query_norm(hidden.to(self.q_proj.weight.dtype)))
        key = self.k_proj(memory + encoding.to(memory.dtype))
        value = self.v_proj(memory)
        def heads(x):
            return x.view(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)
        # Give empty rows a dummy key, then explicitly zero their contribution.
        safe_valid = valid.clone()
        safe_valid[~active, 0] = True
        attended = F.scaled_dot_product_attention(
            heads(query), heads(key), heads(value), attn_mask=safe_valid[:, None, None, :], dropout_p=0.0,
        ).transpose(1, 2).flatten(2)
        delta = self.out_proj(attended)
        delta = delta.masked_fill(~(query_valid & active[:, None])[..., None], 0)
        return hidden + (self.gate.tanh() * delta).to(hidden.dtype)


class HistoryDecoderLayer(nn.Module):
    """Keep native attention/MLP and checkpointing; read history after the whole layer.

    Memory travels as forward arguments, never mutable hooks or shared per-batch state.
    The native decoder can recompute safely during gradient checkpointing.
    """
    def __init__(self, decoder, memory_attention):
        super().__init__()
        self.decoder = decoder
        self.memory_attention = memory_attention
        self.layer_type = decoder.layer_type

    def forward(self, hidden_states, *, memory_tokens=None, memory_valid_mask=None,
                memory_query_mask=None, **kwargs):
        hidden_states = self.decoder(hidden_states, **kwargs)
        if memory_tokens is not None:
            hidden_states = self.memory_attention(hidden_states, memory_tokens, memory_valid_mask, memory_query_mask)
        return hidden_states
