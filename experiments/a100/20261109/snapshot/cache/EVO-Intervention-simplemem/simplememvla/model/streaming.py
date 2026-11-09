"""Inference-only, single-camera streaming with exact-prefix cache validation."""

from functools import partial

import torch
from torch.nn import functional as F

from .modeling_simplemem_reward import SimpleMemRewardOutput, video_groups


def _cached_gdn_chunk(layer, original, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
    """Transformers 5.3 only resumes GDN for seq_len==1; support cached chunks too."""
    if cache_params is None or not cache_params.has_previous_state or hidden_states.shape[1] == 1:
        return original(
            hidden_states, cache_params=cache_params, cache_position=cache_position, attention_mask=attention_mask
        )
    if attention_mask is not None and not attention_mask.bool().all():
        raise ValueError("Cached GDN chunks require unpadded input")
    batch, length, _ = hidden_states.shape
    index = layer.layer_idx
    mixed = layer.in_proj_qkv(hidden_states).transpose(1, 2)
    previous = cache_params.conv_states[index]
    joined = torch.cat([previous, mixed], dim=-1)
    cache_params.conv_states[index] = joined[:, :, -layer.conv_kernel_size :].contiguous()
    context = torch.cat([previous[:, :, -(layer.conv_kernel_size - 1) :], mixed], dim=-1)
    mixed = F.silu(F.conv1d(context, layer.conv1d.weight, layer.conv1d.bias, groups=layer.conv_dim))
    query, key, value = torch.split(mixed.transpose(1, 2), [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, layer.head_k_dim)
    key = key.reshape(batch, length, -1, layer.head_k_dim)
    value = value.reshape(batch, length, -1, layer.head_v_dim)
    repeat = layer.num_v_heads // layer.num_k_heads
    query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
    beta = layer.in_proj_b(hidden_states).sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(layer.in_proj_a(hidden_states).float() + layer.dt_bias)
    output, state = layer.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=cache_params.recurrent_states[index],
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    cache_params.recurrent_states[index] = state
    z = layer.in_proj_z(hidden_states).reshape(-1, layer.head_v_dim)
    output = layer.norm(output.reshape(-1, layer.head_v_dim), z).reshape(batch, length, -1)
    return layer.out_proj(output)


def _fp32_attention(
    layer,
    original,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Avoid SDPA's FP32 GQA math fallback and its quadratic score allocation."""
    if not hidden_states.is_cuda or hidden_states.dtype != torch.float32:
        return original(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    shape = hidden_states.shape[:-1]
    view = (*shape, -1, layer.head_dim)
    query, gate = layer.q_proj(hidden_states).view(*shape, -1, layer.head_dim * 2).chunk(2, dim=-1)
    query = layer.q_norm(query.reshape(view)).transpose(1, 2)
    key = layer.k_norm(layer.k_proj(hidden_states).view(view)).transpose(1, 2)
    value = layer.v_proj(hidden_states).view(view).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    if past_key_values is not None:
        key, value = past_key_values.update(
            key, value, layer.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position}
        )
    key, value = repeat_kv(key, layer.num_key_value_groups), repeat_kv(value, layer.num_key_value_groups)
    if attention_mask is not None and attention_mask.stride(-2) % 8:
        size = attention_mask.shape[-1]
        attention_mask = F.pad(attention_mask, (0, -size % 8))[..., :size]
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=layer.attention_dropout if layer.training else 0.0,
            scale=layer.scaling,
            is_causal=query.shape[-2] > 1 and attention_mask is None,
        )
    output = output.transpose(1, 2).contiguous().reshape(*shape, -1)
    return layer.o_proj(output * gate.reshape(*shape, -1).sigmoid()), None


class SimpleMemRewardStream:
    """Consume processor batches and immutable temporal-patch keys.

    Each key identifies two sampled frames AND their preprocessing/content version.
    Never reuse a key for changed pixels. Appending complete pairs reuses KV/GDN;
    sliding the window or replacing an odd padded pair rebuilds language state.
    Only live-window vision features are retained. Call reset between episodes.
    CPU preprocessing is outside this class and is included in benchmark timings.
    """

    def __init__(self, model):
        self.model = model
        for block in model.backbone.model.language_model.layers:
            attention = getattr(block, "self_attn", None)
            if attention is not None and not getattr(attention, "_simplemem_fp32_attention", False):
                attention.forward = partial(_fp32_attention, attention, attention.forward)
                attention._simplemem_fp32_attention = True
            layer = getattr(block, "linear_attn", None)
            if layer is not None and layer.in_proj_qkv.weight.dtype == torch.float32:
                from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

                layer.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            if layer is not None and not getattr(layer, "_simplemem_cached_chunks", False):
                layer.forward = partial(_cached_gdn_chunk, layer, layer.forward)
                layer._simplemem_cached_chunks = True
        self.reset()

    def reset(self):
        self.features = {}
        self.grid = None
        self.cache = None
        self.ids = self.positions = self.pooled = None
        self.keys = ()
        self.stats = {}

    @torch.inference_mode()
    def step(self, batch, patch_keys):
        if self.model.training:
            raise ValueError("Streaming requires model.eval()")
        keys = tuple(patch_keys)
        ids = batch["input_ids"]
        grid = batch["video_grid_thw"]
        if ids.shape[0] != 1 or grid.shape != (1, 3) or "pixel_values" in batch:
            raise ValueError("Streaming supports one sample, one main-camera video and no current images")
        if not batch["attention_mask"].bool().all():
            raise ValueError("Streaming expects an unpadded sample")
        temporal, height, width = grid[0].tolist()
        if len(keys) != temporal or len(set(keys)) != len(keys):
            raise ValueError("Provide one unique immutable key per temporal patch")
        if self.grid != (height, width):
            self.reset()
            self.grid = (height, width)
        live = set(keys)
        self.features = {key: value for key, value in self.features.items() if key in live}
        qwen = self.model.backbone.model
        device, dtype = self.model.backbone.device, self.model.backbone.dtype
        pixels = batch["pixel_values_videos"].reshape(temporal, height * width, -1)
        missing = [i for i, key in enumerate(keys) if key not in self.features]
        if missing:
            rows = pixels[missing].flatten(0, 1).to(device=device, dtype=dtype)
            grids = torch.tensor([[1, height, width]] * len(missing), device=device)
            encoded = qwen.get_video_features(rows, grids, return_dict=True).pooler_output
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.reshape(len(missing), -1, encoded.shape[-1])
            for index, value in zip(missing, encoded, strict=True):
                self.features[keys[index]] = value
        ids = ids.to(device)
        mask = batch["attention_mask"].to(device)
        grid = grid.to(device)
        types = torch.zeros_like(ids)
        types[ids == qwen.config.video_token_id] = 2
        rope_grid = grid.repeat_interleave(grid[:, 0], dim=0).clone()
        rope_grid[:, 0] = 1
        pos, _ = qwen.get_rope_index(ids, video_grid_thw=rope_grid, attention_mask=mask, mm_token_type_ids=types)
        groups, owners = video_groups(
            ids, mask, grid, qwen.config.video_token_id, qwen.config.vision_config.spatial_merge_size
        )
        # Never cache chat closing tokens: future video tokens must precede them.
        end = int((groups[0] >= 0).nonzero()[-1]) + 1
        ids, pos, groups = ids[:, :end], pos[:, :, :end], groups[:, :end]
        reuse = (
            self.cache is not None
            and keys[: len(self.keys)] == self.keys
            and ids.shape[1] >= self.ids.shape[1]
            and torch.equal(ids[:, : self.ids.shape[1]], self.ids)
            and torch.equal(pos[:, :, : self.ids.shape[1]], self.positions)
        )
        start = self.ids.shape[1] if reuse else 0
        old_groups = len(self.keys) if reuse else 0
        if not reuse:
            self.cache = None
            self.pooled = None
        if start < end:
            embeddings = qwen.get_input_embeddings()(ids[:, start:])
            visual = ids[0, start:] == qwen.config.video_token_id
            embeddings[0, visual] = torch.cat([self.features[key] for key in keys[old_groups:]]).to(dtype)
            cache, self.cache = self.cache, None
            hidden_out = qwen.language_model(
                inputs_embeds=embeddings,
                attention_mask=None,
                position_ids=pos[:, :, start:],
                past_key_values=cache,
                use_cache=True,
            )
            hidden = hidden_out.last_hidden_state[0]
            group_ids = groups[0, start:][visual] - old_groups
            count = temporal - old_groups
            sums = torch.zeros(count, hidden.shape[-1], device=device, dtype=torch.float32).index_add(
                0, group_ids, hidden[visual].float()
            )
            pooled = (sums / torch.bincount(group_ids, minlength=count).unsqueeze(-1)).to(dtype)
            self.pooled = pooled if self.pooled is None else torch.cat([self.pooled, pooled])
            self.cache = hidden_out.past_key_values
        self.ids, self.positions, self.keys = ids.clone(), pos.clone(), keys
        self.stats = {
            "vision_patches": len(missing),
            "language_tokens": end - start,
            "reused_prefix_tokens": start,
            "rebuilt": not reuse,
        }
        pooled = self.pooled[None]
        raw = self.model.progress_head(pooled)
        discrete = self.model.config.progress_loss_type == "discrete"
        if discrete:
            support = torch.linspace(0, 1, raw.shape[-1], device=device)
            progress = (raw.float().softmax(-1) * support).sum(-1)
        else:
            progress = raw.squeeze(-1)
        return SimpleMemRewardOutput(
            progress=progress,
            progress_logits=raw if discrete else None,
            success_logits=self.model.success_head(pooled).squeeze(-1),
            progress_mask=owners >= 0,
            video_indices=owners,
        )
