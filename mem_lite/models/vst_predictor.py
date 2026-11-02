"""Video-XL-style pre-RoPE VST KV memory for the Qwen3.5 hybrid backbone.

Windows are vectorized across frames. Linear layers see independent frames;
full-attention layers see the current frame plus ONLY earlier frames' VSTs.
The forward is functional: checkpoint recomputation cannot mutate a cache.
"""

from dataclasses import dataclass

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

from .m3_predictor import adjacent_bin_targets


@dataclass
class VSTCache:
    # Keys are normalized, BEFORE rotary encoding. Values are unrotated as usual.
    layers: dict
    valid: torch.Tensor

    def detach(self):
        return VSTCache({i: (k.detach(), v.detach()) for i, (k, v) in self.layers.items()}, self.valid.detach())

    def to(self, device):
        return VSTCache({i: (k.to(device), v.to(device)) for i, (k, v) in self.layers.items()}, self.valid.to(device))

    def compact(self):
        """Remove inter-frame padding, preserving time order and autograd links."""
        lengths = self.lengths
        width = int(lengths.max())
        # Sorting a 0/1 mask stably puts real tokens first without reordering them.
        order = (~self.valid).long().argsort(dim=-1, stable=True)[:, :width]
        layers = {}
        for i, (key, value) in self.layers.items():
            index = order[:, None, :, None].expand(-1, key.shape[1], -1, key.shape[3])
            layers[i] = key.gather(2, index), value.gather(2, index)
        valid = torch.arange(width, device=self.valid.device)[None, :] < lengths[:, None]
        return VSTCache(layers, valid)

    @property
    def lengths(self):
        return self.valid.sum(-1)


class VSTAttention(nn.Module):
    """VST-specific Q/K/V with the native O projection shared by all tokens."""

    def __init__(self, native):
        super().__init__()
        self.native = native
        self.vst_projections = nn.ModuleDict()
        for name in ("q_proj", "k_proj", "v_proj"):
            source = getattr(native, name)
            projection = nn.Linear(source.in_features, source.out_features, bias=source.bias is not None,
                                   device=source.weight.device, dtype=torch.float32)
            with torch.no_grad():
                projection.weight.copy_(source.weight)
                if source.bias is not None:
                    projection.bias.copy_(source.bias)
            self.vst_projections[name] = projection

    def project(self, name, states, flat_indices):
        ordinary = getattr(self.native, name)(states)
        special = self.vst_projections[name](states.flatten(0, 1).index_select(0, flat_indices))
        return ordinary.flatten(0, 1).index_copy(0, flat_indices, special.to(ordinary.dtype)).view_as(ordinary)

    def forward(self, hidden, old_key, old_value, *, batch, window, vst_indices, flat_indices,
                current_rotary, memory_rotary, attention_mask, memory_valid):
        n, seq, _ = hidden.shape
        dim, heads = self.native.head_dim, self.native.config.num_key_value_heads
        query, gate = self.project("q_proj", hidden, flat_indices).view(n, seq, -1, dim * 2).chunk(2, dim=-1)
        query = self.native.q_norm(query).transpose(1, 2)
        key = self.native.k_norm(self.project("k_proj", hidden, flat_indices).view(n, seq, heads, dim)).transpose(1, 2)
        value = self.project("v_proj", hidden, flat_indices).view(n, seq, heads, dim).transpose(1, 2)
        index = vst_indices[:, None, :, None].expand(-1, heads, -1, dim)
        def memory_slots(states):
            return states.gather(2, index).view(batch, window, heads, -1, dim).transpose(1, 2).flatten(2, 3)
        new_key, new_value = memory_slots(key), memory_slots(value)
        memory_key = torch.cat((old_key, new_key), dim=2).masked_fill(~memory_valid[:, None, :, None], 0)
        memory_value = torch.cat((old_value, new_value), dim=2).masked_fill(~memory_valid[:, None, :, None], 0)
        query, key = apply_rotary_pos_emb(query, key, *current_rotary)
        # All history, including earlier frames in this window, is reindexed in
        # compact 1-D VST positions. No already-rotated key is cached or rotated twice.
        _, rotated_memory = apply_rotary_pos_emb(memory_key, memory_key, *memory_rotary)
        def expand_frames(states):
            return states[:, None].expand(-1, window, -1, -1, -1).reshape(n, heads, -1, dim)
        keys = torch.cat((expand_frames(rotated_memory), key), dim=2)
        values = torch.cat((expand_frames(memory_value), value), dim=2)
        extra = attention_mask.shape[-1] - keys.shape[-2]
        if extra:
            keys, values = F.pad(keys, (0, 0, 0, extra)), F.pad(values, (0, 0, 0, extra))
        # Explicit GQA expansion keeps memory-efficient SDPA available on A100.
        keys, values = repeat_kv(keys, self.native.num_key_value_groups), repeat_kv(values, self.native.num_key_value_groups)
        attended = F.scaled_dot_product_attention(
            query, keys, values, attn_mask=attention_mask,
            dropout_p=self.native.attention_dropout if self.training else 0.0, scale=self.native.scaling,
        ).transpose(1, 2).reshape(n, seq, -1)
        attended = attended * gate.reshape(n, seq, -1).sigmoid()
        # Video-XL's released configuration specializes Q/K/V only. All tokens
        # use the same native output projection, including its language LoRA.
        return self.native.o_proj(attended), memory_key, memory_value


class VSTProgressPredictor(nn.Module):
    def __init__(self, model_path, num_bins=101, dtype=torch.bfloat16, lora_rank=16,
                 lora_alpha=32, lora_dropout=0.05, gradient_checkpointing=True):
        super().__init__()
        if gradient_checkpointing not in (True, False, "full_attention"):
            raise ValueError("gradient_checkpointing must be bool or 'full_attention'")
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa", local_files_only=True,
        )
        self.qwen.requires_grad_(False)
        targets = [name for name, module in self.qwen.named_modules()
                   if name.startswith("model.language_model.layers.") and isinstance(module, nn.Linear)]
        inject_adapter_in_model(LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                          target_modules=targets, bias="none"), self.qwen)
        for parameter in self.qwen.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        self.full_layers = []
        for i, layer in enumerate(self.qwen.model.language_model.layers):
            if layer.layer_type == "full_attention":
                layer.self_attn = VSTAttention(layer.self_attn)
                self.full_layers.append(i)
        self.gradient_checkpointing = gradient_checkpointing
        hidden = self.qwen.config.text_config.hidden_size
        self.vst_embedding = nn.Parameter(torch.empty(hidden, dtype=torch.float32))
        nn.init.normal_(self.vst_embedding, std=self.qwen.config.text_config.initializer_range)
        self.progress_head = nn.Linear(hidden, num_bins)
        self.register_buffer("bin_centers", torch.linspace(0, 1, num_bins))
        self.qwen.model.visual.eval()

    def train(self, mode=True):
        super().train(mode)
        self.qwen.model.visual.eval()
        return self

    def forward(self, batch, target_pt=None, cache=None):
        with torch.autocast(self.qwen.device.type, dtype=self.qwen.dtype, enabled=self.qwen.dtype != torch.float32):
            return self._forward(batch, target_pt, cache)

    def _forward(self, batch, target_pt, cache):
        ids, valid = batch["input_ids"], batch["attention_mask"].bool()
        b, w = batch["stream_batch"], batch["window_size"]
        n, seq = ids.shape
        if n != b * w:
            raise ValueError("Invalid stream/time dimensions")
        text = self.qwen.model.language_model
        hidden = text.embed_tokens(ids)
        with torch.no_grad():
            features = self.qwen.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"]).pooler_output
            features = torch.cat(features, dim=0).to(hidden.dtype)
        image_mask = (ids == self.qwen.config.image_token_id).unsqueeze(-1).expand_as(hidden)
        if int(image_mask.sum()) != features.numel():
            raise ValueError("Image features and placeholders disagree")
        hidden = hidden.masked_scatter(image_mask, features)
        hidden = torch.where(batch["vst_mask"][..., None], self.vst_embedding.to(hidden.dtype), hidden)
        slot_valid = (batch["vst_valid"] & batch["frame_valid"][:, None]).view(b, w, -1)
        m = slot_valid.shape[-1]
        if cache is None:
            old_valid = torch.zeros(b, 0, dtype=torch.bool, device=ids.device)
        else:
            old_valid = cache.valid
            if old_valid.shape[0] != b or set(cache.layers) != set(self.full_layers):
                raise ValueError("Cache batch/layers disagree; reset on episode changes")
        p = old_valid.shape[1]
        memory_valid = torch.cat((old_valid, slot_valid.flatten(1)), dim=1)
        before = slot_valid.sum(-1).cumsum(-1) - slot_valid.sum(-1) + old_valid.sum(-1)[:, None]
        positions = batch["position_ids"] + before.reshape(1, n, 1)
        current_rotary = text.rotary_emb(hidden, positions)
        memory_positions = (memory_valid.long().cumsum(-1) - 1).clamp_min(0)
        memory_rotary = text.rotary_emb(hidden, memory_positions[None].expand(3, -1, -1))
        # Shared prefix + previous-frame VSTs + own frame's causal tokens.
        previous_frame = torch.arange(w, device=ids.device)[None, :] < torch.arange(w, device=ids.device)[:, None]
        allowed_new = slot_valid[:, None].expand(-1, w, -1, -1) & previous_frame[None, :, :, None]
        allowed_history = torch.cat((old_valid[:, None].expand(-1, w, -1), allowed_new.flatten(2)), dim=-1)
        allowed_history = allowed_history.reshape(n, 1, 1, -1).expand(-1, 1, seq, -1)
        causal = torch.ones(seq, seq, dtype=torch.bool, device=ids.device).tril()
        allowed_current = causal[None, None] & valid[:, None, None, :]
        attention_mask = torch.cat((allowed_history, allowed_current), dim=-1)
        attention_mask = F.pad(attention_mask, (0, (-attention_mask.shape[-1]) % 8))
        new_layers = {}
        for i, layer in enumerate(text.layers):
            if layer.layer_type == "linear_attention":
                # No native cache: convolution and recurrent state reset per frame.
                def run_linear(states, decoder=layer):
                    return decoder(states, position_embeddings=current_rotary, attention_mask=valid,
                                   past_key_values=None, use_cache=False)
                hidden = (checkpoint(run_linear, hidden, use_reentrant=False)
                          if self.training and self.gradient_checkpointing is True else run_linear(hidden))
            else:
                dim = self.qwen.config.text_config.head_dim
                heads = self.qwen.config.text_config.num_key_value_heads
                old_key, old_value = (cache.layers[i] if cache is not None else
                                      (hidden.new_empty(b, heads, 0, dim), hidden.new_empty(b, heads, 0, dim)))
                def run_full(states, past_k, past_v, decoder=layer):
                    attended, next_k, next_v = decoder.self_attn(
                        decoder.input_layernorm(states), past_k, past_v, batch=b, window=w,
                        vst_indices=batch["vst_indices"], flat_indices=batch["vst_flat_indices"],
                        current_rotary=current_rotary, memory_rotary=memory_rotary,
                        attention_mask=attention_mask, memory_valid=memory_valid,
                    )
                    states = states + attended
                    return states + decoder.mlp(decoder.post_attention_layernorm(states)), next_k, next_v
                hidden, key, value = (checkpoint(run_full, hidden, old_key, old_value, use_reentrant=False)
                                      if self.training and self.gradient_checkpointing else run_full(hidden, old_key, old_value))
                new_layers[i] = (key, value)
        hidden = text.norm(hidden)
        last = valid.long().sum(-1) - 1  # collator uses right padding
        pooled = hidden[torch.arange(n, device=ids.device), last]
        logits = self.progress_head(pooled.to(self.progress_head.weight.dtype)).float()
        probabilities = logits.softmax(-1)
        result = {"pt": (probabilities * self.bin_centers.float()).sum(-1), "bin_logits": logits,
                  "bin_probs": probabilities, "cache": VSTCache(new_layers, memory_valid).compact()}
        if target_pt is not None:
            targets = adjacent_bin_targets(target_pt.to(ids.device), len(self.bin_centers))
            weights = batch["frame_valid"].to(logits.dtype)
            loss_sum = (-(targets * logits.log_softmax(-1)).sum(-1) * weights).sum()
            result.update(loss_sum=loss_sum, loss=loss_sum / weights.sum().clamp_min(1), num_samples=weights.sum())
        return result
