"""Robometer heads adapted to SimpleMem's native Qwen3.5 video context.

Head architecture: robometer/robometer @ 352d160389daa964788de1ec933d1925f3a6de4f,
robometer/models/heads.py. See ../ROBOMETER_SOURCE.md for attribution and differences.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_simplemem_reward import SimpleMemRewardConfig


def prediction_head(width, outputs, dropout, sigmoid=False):
    layers = [
        nn.Linear(width, width // 2),
        nn.LayerNorm(width // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(width // 2, outputs),
    ]
    if sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def video_groups(input_ids, attention_mask, video_grid_thw, video_token_id, merge_size):
    """Map only video tokens to temporal groups, skipping timestamps and wrist images.

    Returns per-token group IDs (-1 elsewhere) and per-group video indices (-1 padding).
    Each temporal group represents one native temporal patch, normally two sampled frames.
    """
    groups = torch.full_like(input_ids, -1)
    videos = []
    grid_index = 0
    grids = video_grid_thw.tolist()
    for batch, ids in enumerate(input_ids):
        positions = ((ids == video_token_id) & attention_mask[batch].bool()).nonzero().flatten()
        offset, group = 0, 0
        owners = []
        video_index = 0
        while offset < len(positions):
            if grid_index >= len(grids):
                raise ValueError("Video tokens exceed video_grid_thw")
            temporal, height, width = grids[grid_index]
            if min(temporal, height, width) <= 0 or height % merge_size or width % merge_size:
                raise ValueError("Invalid video grid for spatial merge size")
            per_step = height * width // merge_size**2
            count = temporal * per_step
            if offset + count > len(positions):
                raise ValueError("Video grid exceeds video tokens in sample")
            selected = positions[offset : offset + count]
            groups[batch, selected] = torch.arange(group, group + temporal, device=ids.device).repeat_interleave(
                per_step
            )
            owners.extend([video_index] * temporal)
            offset += count
            group += temporal
            video_index += 1
            grid_index += 1
        if not owners:
            raise ValueError("Each sample requires native video tokens")
        videos.append(torch.tensor(owners, device=ids.device))
    if grid_index != len(grids):
        raise ValueError("Unused video grids")
    return groups, pad_sequence(videos, batch_first=True, padding_value=-1)


@dataclass
class SimpleMemRewardOutput(ModelOutput):
    loss: torch.Tensor | None = None
    progress_loss: torch.Tensor | None = None
    success_loss: torch.Tensor | None = None
    preference_loss: torch.Tensor | None = None
    progress: torch.Tensor | None = None
    progress_logits: torch.Tensor | None = None
    success_logits: torch.Tensor | None = None
    preference_logits: torch.Tensor | None = None
    progress_mask: torch.Tensor | None = None
    video_indices: torch.Tensor | None = None


class SimpleMemForRewardPrediction(PreTrainedModel):
    config_class = SimpleMemRewardConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def __init__(self, config, backbone=None):
        super().__init__(config)
        if backbone is None:
            backbone_config = (
                AutoConfig.for_model(**dict(config.backbone_config))
                if config.backbone_config
                else AutoConfig.from_pretrained(config.backbone_model_name_or_path)
            )
            backbone = AutoModelForImageTextToText.from_config(backbone_config)
        self.backbone = backbone
        config.backbone_config = backbone.config.to_dict()
        width = backbone.config.text_config.hidden_size
        discrete = config.progress_loss_type == "discrete"
        self.progress_head = prediction_head(
            width, config.progress_discrete_bins if discrete else 1, config.head_dropout, sigmoid=not discrete
        )
        self.success_head = prediction_head(width, 1, config.head_dropout)
        self.preference_head = prediction_head(width, 1, config.head_dropout)
        for parameter in backbone.parameters():
            parameter.requires_grad_(not config.freeze_backbone)
        for head in (self.progress_head, self.success_head, self.preference_head):
            head.to(device=backbone.device, dtype=backbone.dtype)
        # HF preserves initialized backbone weights and restores nonpersistent RoPE buffers on reload.
        self.post_init()

    @classmethod
    def from_backbone(cls, path, config=None, **kwargs):
        config = config or SimpleMemRewardConfig()
        config.backbone_model_name_or_path = path
        return cls(config, AutoModelForImageTextToText.from_pretrained(path, **kwargs))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def prepare_processor(self, processor):
        """Register the upstream preference token before training; save processor with model."""
        tokenizer = processor.tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|pref_token|>"]})
        self.backbone.resize_token_embeddings(len(tokenizer))
        self.config.preference_token_id = tokenizer.convert_tokens_to_ids("<|pref_token|>")
        self.config.backbone_config = self.backbone.config.to_dict()

    def forward(
        self,
        input_ids,
        attention_mask,
        video_grid_thw,
        progress_targets=None,
        success_targets=None,
        preference_targets=None,
        **kwargs,
    ):
        # Qwen3.5 native video has a separate timestamp-delimited span per temporal patch.
        # Transformers 5.3 expects one grid per span for RoPE, but the ViT still needs
        # the original full-video grids. Expanding only the RoPE input also works
        # with versions that perform the temporal expansion internally (T is now 1).
        if "position_ids" not in kwargs:
            types = kwargs.get("mm_token_type_ids")
            if types is None:
                types = torch.zeros_like(input_ids)
                types[input_ids == self.backbone.config.image_token_id] = 1
                types[input_ids == self.backbone.config.video_token_id] = 2
            rope_grids = video_grid_thw.repeat_interleave(video_grid_thw[:, 0], dim=0).clone()
            rope_grids[:, 0] = 1
            kwargs["position_ids"], _ = self.backbone.model.get_rope_index(
                input_ids,
                image_grid_thw=kwargs.get("image_grid_thw"),
                video_grid_thw=rope_grids,
                attention_mask=attention_mask,
                mm_token_type_ids=types,
            )
        hidden = self.backbone.model(
            input_ids=input_ids, attention_mask=attention_mask, video_grid_thw=video_grid_thw, use_cache=False, **kwargs
        ).last_hidden_state
        groups, owners = video_groups(
            input_ids,
            attention_mask,
            video_grid_thw,
            self.backbone.config.video_token_id,
            self.backbone.config.vision_config.spatial_merge_size,
        )
        pooled = []
        for batch in range(len(input_ids)):
            valid = groups[batch] >= 0
            ids = groups[batch, valid]
            count = int(ids.max()) + 1
            sums = torch.zeros(count, hidden.shape[-1], device=hidden.device, dtype=torch.float32).index_add(
                0, ids, hidden[batch, valid].float()
            )
            pooled.append((sums / torch.bincount(ids, minlength=count).unsqueeze(-1)).to(hidden.dtype))
        pooled = pad_sequence(pooled, batch_first=True)
        valid = owners >= 0
        raw_progress = self.progress_head(pooled)
        discrete = self.config.progress_loss_type == "discrete"
        if discrete:
            support = torch.linspace(0, 1, self.config.progress_discrete_bins, device=hidden.device)
            progress = (raw_progress.float().softmax(-1) * support).sum(-1)
        else:
            progress = raw_progress.squeeze(-1)
        success = self.success_head(pooled).squeeze(-1)
        pref_logits = None
        if self.config.preference_token_id is not None:
            pref_mask = (input_ids == self.config.preference_token_id) & attention_mask.bool()
            if pref_mask.any():
                if not torch.all(pref_mask.sum(-1) == 1):
                    raise ValueError("Preference batches require exactly one preference token per sample")
                visual = (
                    (input_ids == self.backbone.config.video_token_id)
                    | (input_ids == self.backbone.config.image_token_id)
                ) & attention_mask.bool()
                positions = torch.arange(input_ids.shape[1], device=input_ids.device)
                last_visual = positions.expand_as(input_ids).masked_fill(~visual, -1).amax(-1)
                pref_positions = pref_mask.long().argmax(-1)
                if not torch.all(owners.amax(-1) == 1) or not torch.all(pref_positions > last_visual):
                    raise ValueError("Preference requires two history videos and a token after all visual input")
                pref_logits = self.preference_head(hidden[pref_mask]).squeeze(-1)

        progress_loss = success_loss = preference_loss = None
        if progress_targets is not None:
            target = progress_targets.to(hidden.device)
            if target.shape != valid.shape:
                raise ValueError("progress_targets must match temporal groups [batch, max_groups]; -100 means ignore")
            mask = valid & (target != -100)
            if mask.any():
                selected = target[mask]
                if not torch.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
                    raise ValueError("Progress targets must be finite values in [0, 1]")
                if discrete:
                    # C51 interpolation between adjacent support bins, as supported by Robometer.
                    location = selected.float() * (self.config.progress_discrete_bins - 1)
                    lower = location.floor().long()
                    upper = (lower + 1).clamp_max(self.config.progress_discrete_bins - 1)
                    fraction = location - lower
                    log_probs = raw_progress[mask].float().log_softmax(-1)
                    progress_loss = (
                        -(1 - fraction) * log_probs.gather(1, lower[:, None]).squeeze(1)
                        - fraction * log_probs.gather(1, upper[:, None]).squeeze(1)
                    ).mean()
                else:
                    loss_fn = F.l1_loss if self.config.progress_loss_type == "l1" else F.mse_loss
                    progress_loss = loss_fn(progress[mask].float(), selected.float())
            else:
                progress_loss = raw_progress.sum() * 0
        if success_targets is not None:
            target = success_targets.to(hidden.device)
            if target.shape != valid.shape:
                raise ValueError("success_targets must match temporal groups; -100 means ignore")
            mask = valid & (target != -100)
            if mask.any():
                selected = target[mask].float()
                if not torch.all((selected == 0) | (selected == 1)):
                    raise ValueError("Success targets must be 0 or 1")
                # Balance classes when both are present; single-class batches retain unit weight.
                positives = selected.sum()
                negatives = len(selected) - positives
                weights = torch.ones_like(selected)
                if positives > 0 and negatives > 0:
                    weights = torch.where(
                        selected.bool(), len(selected) / (2 * positives), len(selected) / (2 * negatives)
                    )
                success_loss = F.binary_cross_entropy_with_logits(success[mask].float(), selected, weight=weights)
            else:
                success_loss = success.sum() * 0
        if preference_targets is not None:
            if pref_logits is None:
                raise ValueError("Preference labels require a registered <|pref_token|> after both trajectories")
            target = preference_targets.to(device=hidden.device, dtype=torch.float32)
            if target.shape != pref_logits.shape or not torch.all((target == 0) | (target == 1)):
                raise ValueError("preference_targets must be [batch], 1 for A preferred and 0 for B")
            preference_loss = F.binary_cross_entropy_with_logits(pref_logits.float().clamp(-50, 50), target)
        losses = [loss for loss in (progress_loss, success_loss, preference_loss) if loss is not None]
        return SimpleMemRewardOutput(
            loss=sum(losses) if losses else None,
            progress_loss=progress_loss,
            success_loss=success_loss,
            preference_loss=preference_loss,
            progress=progress.masked_fill(~valid, 0),
            progress_logits=raw_progress if discrete else None,
            success_logits=success,
            preference_logits=pref_logits,
            progress_mask=valid,
            video_indices=owners,
        )
