"""Temporal intervention-probability predictor for Pi05 VLM features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn


class _QueryPool(nn.Module):
    """Pool a variable number of tokens into a fixed number of learned tokens."""

    def __init__(self, input_dim: int, output_dim: int, num_queries: int, num_heads: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.queries = nn.Parameter(torch.randn(num_queries, output_dim) * output_dim**-0.5)
        self.attention = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: [batch, token_count, input_dim]
        values = self.input_proj(tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attention(queries, values, values, need_weights=False)
        return self.norm(queries + pooled)


@dataclass(frozen=True)
class InterventionPredictorConfig:
    hidden_dim: int
    image_input_dim: int
    state_input_dim: int
    language_input_dim: int
    num_layers: int
    num_image_queries: int = 8
    num_state_queries: int = 2
    num_layer_queries: int = 16
    temporal_layers: int = 3
    temporal_heads: int = 8
    short_window: int = 3
    dropout: float = 0.1


class InterventionPredictor(nn.Module):
    """Predict intervention probability from a temporal window of Pi05 features.

    Expected inputs:
      language_tokens: [B, language_token_count, language_input_dim]
      image_tokens: [B, N, L, K, image_input_dim]
      state_tokens: [B, N, L, H, state_input_dim]

    ``L`` is the number of selected VLM layers. The model returns a dictionary
    containing both the training logit and its sigmoid probability.
    """

    def __init__(self, config: InterventionPredictorConfig) -> None:
        super().__init__()
        if config.temporal_heads <= 0 or config.hidden_dim % config.temporal_heads != 0:
            raise ValueError("hidden_dim must be divisible by temporal_heads")
        if config.short_window <= 0:
            raise ValueError("short_window must be positive")

        self.config = config
        self.image_pool = _QueryPool(
            config.image_input_dim, config.hidden_dim, config.num_image_queries, config.temporal_heads
        )
        self.state_pool = _QueryPool(
            config.state_input_dim, config.hidden_dim, config.num_state_queries, config.temporal_heads
        )
        self.layer_pool = _QueryPool(
            config.hidden_dim, config.hidden_dim, config.num_layer_queries, config.temporal_heads
        )
        self.language_proj = nn.Linear(config.language_input_dim, config.hidden_dim)
        self.language_norm = nn.LayerNorm(config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.temporal_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, config.temporal_layers)
        self.temporal_norm = nn.LayerNorm(config.hidden_dim)
        self.temporal_pos = nn.Parameter(torch.randn(1, 256, config.hidden_dim) * config.hidden_dim**-0.5)

        self.output_head = nn.Sequential(
            nn.LayerNorm(3 * config.hidden_dim),
            nn.Linear(3 * config.hidden_dim, 2 * config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.hidden_dim, 1),
        )

    def forward(
        self,
        language_tokens: Tensor,
        image_tokens: Tensor,
        state_tokens: Tensor,
    ) -> dict[str, Tensor]:
        if image_tokens.ndim != 5 or state_tokens.ndim != 5 or language_tokens.ndim != 3:
            raise ValueError("Expected language [B,S,D], image [B,N,L,K,D], and state [B,N,L,H,D]")
        batch, frames, layers, _, _ = image_tokens.shape
        if state_tokens.shape[:3] != (batch, frames, layers):
            raise ValueError("image_tokens and state_tokens must agree on batch, frames, and layers")
        if layers != self.config.num_layers:
            raise ValueError(f"Expected {self.config.num_layers} selected layers, got {layers}")
        if frames == 0 or layers == 0:
            raise ValueError("The temporal window and selected layer list cannot be empty")

        image = image_tokens.reshape(batch * frames * layers, *image_tokens.shape[3:]) # 每张图像的TOKEN数量减少到num_image_queries
        state = state_tokens.reshape(batch * frames * layers, *state_tokens.shape[3:])
        image = self.image_pool(image)
        state = self.state_pool(state)
        per_layer = torch.cat((image, state), dim=1) # 这里进行了跨层拼接
        # image 的形状是 (batch * frames * layers, num_image_queries, hidden_dim)
        # state 的形状是 (batch * frames * layers, num_state_queries, hidden_dim)。
        # 拼接后 per_layer 的形状是 (batch * frames * layers, num_image_queries + num_state_queries, hidden_dim)。
        per_frame = self.layer_pool(per_layer.reshape(batch * frames, layers * per_layer.shape[1], -1))
        # (batch * frames, layers * Q, hidden_dim) - > (batch * frames, num_layer_queries, hidden_dim)
        # 这里做了layer级别的池化
        language = self.language_proj(language_tokens).mean(dim=1)
        language = self.language_norm(language)
        temporal = per_frame.reshape(batch, frames * self.config.num_layer_queries, -1)
        temporal = temporal + language[:, None, :] # (batch, num_layer_queries * frames, hidden_dim) 广播加法
        if temporal.shape[1] > self.temporal_pos.shape[1]:
            raise ValueError("The frame window is too large for the temporal positional embedding")
        temporal = temporal + self.temporal_pos[:, : temporal.shape[1]] # (batch, num_layer_queries * frames, hidden_dim)

        # Explicitly expose short-term change information to the temporal encoder.
        temporal = temporal.reshape(batch, frames, self.config.num_layer_queries, -1) # (batch, num_layer_queries * frames, hidden_dim)
        delta = torch.zeros_like(temporal)
        if frames > 1:
            delta[:, 1:] = temporal[:, 1:] - temporal[:, :-1]
        temporal = (temporal + delta).reshape(batch, -1, temporal.shape[-1])
        temporal = self.temporal_norm(self.temporal_encoder(temporal))
        temporal = temporal.reshape(batch, frames, self.config.num_layer_queries, -1)

        latest = temporal[:, -1].mean(dim=1)
        short = temporal[:, -min(frames, self.config.short_window) :].mean(dim=(1, 2))
        long = temporal.mean(dim=(1, 2))
        logits = self.output_head(torch.cat((latest, short, long), dim=-1)).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits)}
