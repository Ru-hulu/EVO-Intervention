"""V-JEPA 2-AC world model: frozen JEPA encoder + action-conditioned latent predictor.

Latents are per-frame patch grids `[B, T, N, D]` (N = tokens_per_frame). Each frame is encoded
on its own by repeating it `tubelet_size` times, as in upstream AC training and planning.
The predictor output at step t is the prediction of the latent at step t + 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from evo_intervention.model.modules.world_model.action_contract import StateTransition
from evo_intervention.model.modules.world_model.vjepa2 import vision_transformer
from evo_intervention.model.modules.world_model.vjepa2.ac_predictor import vit_ac_predictor

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CheckpointReport:
    path: str
    key: str
    loaded: int
    missing: list[str]
    unexpected: list[str]
    shape_mismatch: list[str]


def _clean_state_dict(state_dict: dict) -> dict:
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


def load_checkpoint_part(module: nn.Module, path: str | Path, key: str, reinit_keys: Sequence[str] = ()):
    """Strictly load `checkpoint[key]` into `module`.

    Every parameter must be loaded with a matching shape, except those whose name starts with a
    prefix in `reinit_keys` (e.g. `action_encoder` for a different action width): those may be
    missing or shape-mismatched and keep their fresh initialization. Unexpected keys are errors.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if key not in checkpoint:
        raise KeyError(f"{path} has no '{key}' entry; available: {sorted(checkpoint)}")
    state_dict = _clean_state_dict(checkpoint[key])
    own = module.state_dict()

    def reinit(k):
        return any(k == p or k.startswith(p + ".") for p in reinit_keys)

    mismatch = [k for k, v in state_dict.items() if k in own and own[k].shape != v.shape]
    unexpected = [k for k in state_dict if k not in own]
    missing = [k for k in own if k not in state_dict]
    errors = {
        "shape mismatch": [k for k in mismatch if not reinit(k)],
        "missing": [k for k in missing if not reinit(k)],
        "unexpected": unexpected,
    }
    errors = {name: keys for name, keys in errors.items() if keys}
    if errors:
        raise ValueError(f"Cannot load '{key}' from {path} strictly: {errors}")
    for k in mismatch:
        state_dict.pop(k)
    if not state_dict:
        raise ValueError(f"'{key}' in {path} contains no loadable tensors")
    module.load_state_dict(state_dict, strict=False)  # only reinit_keys can be absent at this point
    report = CheckpointReport(
        path=str(path),
        key=key,
        loaded=len(state_dict),
        missing=list(missing),
        unexpected=[],
        shape_mismatch=mismatch,
    )
    logger.info(
        "Loaded %s/%s: %d tensors; re-initialized %d missing and %d shape-mismatched (reinit_keys=%s)",
        path,
        key,
        report.loaded,
        len(report.missing),
        len(report.shape_mismatch),
        list(reinit_keys),
    )
    return report


def latent_l1(pred: torch.Tensor, target: torch.Tensor, loss_exp: float = 1.0) -> torch.Tensor:
    """Upstream latent loss, reduced over patches and channels: [..., N, D] -> [...]."""
    return (pred - target).abs().pow(loss_exp).mean(dim=(-2, -1)) / loss_exp


class VJEPA2ACWorldModel(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: nn.Module, normalize_reps: bool = True):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.normalize_reps = normalize_reps
        self.img_size = (encoder.img_height, encoder.img_width)
        self.tubelet_size = encoder.tubelet_size
        self.tokens_per_frame = predictor.grid_height * predictor.grid_width
        self.max_steps = predictor.num_frames // predictor.tubelet_size

    @classmethod
    def from_config(cls, cfg) -> "VJEPA2ACWorldModel":
        """Build from the `world_model` config section (see config/world_model/vjepa2_ac_vitg.yaml)."""
        img_size = (cfg.img_size, cfg.img_size) if isinstance(cfg.img_size, int) else tuple(cfg.img_size)
        common = dict(
            img_size=img_size, patch_size=cfg.patch_size, num_frames=cfg.num_frames, tubelet_size=cfg.tubelet_size
        )
        encoder = vision_transformer.__dict__[cfg.encoder.arch](
            **common,
            use_sdpa=True,
            uniform_power=False,
            use_rope=True,
            use_activation_checkpointing=cfg.encoder.get("activation_checkpointing", False),
            **cfg.encoder.get("kwargs", {}),
        )
        predictor = vit_ac_predictor(
            **common,
            embed_dim=encoder.embed_dim,
            predictor_embed_dim=cfg.predictor.embed_dim,
            depth=cfg.predictor.depth,
            num_heads=cfg.predictor.num_heads,
            action_embed_dim=cfg.predictor.action_dim,
            is_frame_causal=True,
            use_rope=True,
            uniform_power=True,
            use_activation_checkpointing=cfg.predictor.get("activation_checkpointing", False),
        )
        if not predictor.use_extrinsics:  # unused parameters would break DDP
            predictor.extrinsics_encoder.requires_grad_(False)
        model = cls(encoder, predictor, normalize_reps=cfg.get("normalize_reps", True))

        ckpt = cfg.get("checkpoint", {})
        model.load_reports = []
        if ckpt.get("encoder_path"):
            model.load_reports.append(
                load_checkpoint_part(encoder, ckpt.encoder_path, ckpt.get("encoder_key", "encoder"))
            )
        if ckpt.get("predictor_path"):
            model.load_reports.append(
                load_checkpoint_part(
                    predictor,
                    ckpt.predictor_path,
                    ckpt.get("predictor_key", "predictor"),
                    reinit_keys=list(ckpt.get("predictor_reinit_keys", [])),
                )
            )
        return model

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        return self

    # ------------------------------------------------------------------ encoding

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """uint8 or [0, 1] float frames [B, T, 3, H, W] -> resized, ImageNet-normalized float."""
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        B, T = frames.shape[:2]
        x = frames.flatten(0, 1)
        if tuple(x.shape[-2:]) != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False, antialias=True)
        mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return ((x - mean) / std).unflatten(0, (B, T))

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Preprocessed frames [B, T, 3, H, W] -> latents [B, T, N, D]. Gradients follow the caller."""
        B, T = frames.shape[:2]
        clips = frames.flatten(0, 1).unsqueeze(2).repeat(1, 1, self.tubelet_size, 1, 1)  # [B*T, 3, tubelet, H, W]
        h = self.encoder(clips)
        h = h.view(B, T, -1, h.size(-1))
        return self._norm(h)

    def _norm(self, h: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(h, (h.size(-1),)) if self.normalize_reps else h

    # ---------------------------------------------------------------- prediction

    def predict(self, latents: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Teacher-forced one-step predictions for every input step.

        latents [B, T, N, D], actions [B, T, A], states [B, T, S] -> [B, T, N, D], where output t
        predicts the latent at t + 1 from inputs up to t.
        """
        B, T, N, _ = latents.shape
        if actions.shape[:2] != (B, T) or states.shape[:2] != (B, T):
            raise ValueError(f"Expected actions/states with leading dims {(B, T)}, got {actions.shape} / {states.shape}")
        if N != self.tokens_per_frame:
            raise ValueError(f"Expected {self.tokens_per_frame} tokens per frame, got {N}")
        out = self.predictor(latents.flatten(1, 2), actions, states)
        return self._norm(out.view(B, T, N, -1))

    def rollout(
        self,
        context_latents: torch.Tensor,
        context_states: torch.Tensor,
        context_actions: torch.Tensor,
        future_actions: torch.Tensor,
        state_transition: Optional[StateTransition] = None,
        future_states: Optional[torch.Tensor] = None,
        transition_actions: Optional[torch.Tensor] = None,
        truncate_every: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive latent rollout; predicted latents are fed back, true future latents never are.

        context_latents    [B, Tc, N, D]  observed latents, the last one is "now"
        context_states     [B, Tc, S]     observed states for those latents
        context_actions    [B, Tc-1, A]   executed latent actions between context latents
        future_actions     [B, H, A]      latent actions to evaluate, starting from "now"
        future_states      [B, H-1, S]    recorded states after each future step (training only);
                                          otherwise `state_transition` produces them
        transition_actions [B, H, k, A']  raw control actions of each latent step for `state_transition`;
                                          defaults to the latent action (k = 1). Needed whenever the
                                          latent action aggregates several control actions.
        truncate_every                    truncated backprop: every this many steps the fed-back history is
                                          detached, so gradients flow only within blocks of that length.
                                          This bounds gradient paths, not forward activation memory.

        Returns predicted latents [B, H, N, D] and the state paired with each predicted latent
        [B, H, S]; with recorded states only, the final state is unknown and H-1 are returned.
        """
        Tc, H = context_latents.shape[1], future_actions.shape[1]
        if context_actions.shape[1] != Tc - 1:
            raise ValueError(f"Need {Tc - 1} context actions for {Tc} context latents, got {context_actions.shape[1]}")
        if Tc + H - 1 > self.max_steps:
            raise ValueError(f"Rollout needs {Tc + H - 1} predictor steps, capacity is {self.max_steps}")
        if state_transition is None:
            num_states = 0 if future_states is None else future_states.shape[1]
            if num_states < H - 1:
                raise ValueError(
                    f"{H}-step rollout without a state_transition needs {H - 1} future_states, got {num_states}"
                )
        if transition_actions is None:
            transition_actions = future_actions.unsqueeze(2)
        if transition_actions.shape[1] != H:
            raise ValueError(f"transition_actions must have {H} steps, got {transition_actions.shape[1]}")
        if truncate_every is not None and truncate_every < 1:
            raise ValueError("truncate_every must be >= 1")

        states = context_states
        actions = torch.cat((context_actions, future_actions[:, :1]), dim=1)
        history, predictions, next_states = [], [], []
        for h in range(H):
            if truncate_every is not None and h > 0 and h % truncate_every == 0:
                history = [x.detach() for x in history]
            latents = torch.cat((context_latents, *history), dim=1)
            prediction = self.predict(latents, actions, states)[:, -1:]
            predictions.append(prediction)
            history.append(prediction)
            if future_states is not None and h < future_states.shape[1]:
                next_state = future_states[:, h : h + 1]
            elif state_transition is not None:
                next_state = state_transition(states[:, -1], transition_actions[:, h]).unsqueeze(1)
            else:  # last step without a transition model: nothing to extrapolate
                break
            next_states.append(next_state)
            if h + 1 == H:
                break
            states = torch.cat((states, next_state), dim=1)
            actions = torch.cat((actions, future_actions[:, h + 1 : h + 2]), dim=1)
        return torch.cat(predictions, dim=1), torch.cat(next_states, dim=1) if next_states else states[:, :0]


def teacher_forcing_rollout_loss(
    world_model: VJEPA2ACWorldModel,
    target_latents: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    rollout_steps: int,
    valid: Optional[torch.Tensor] = None,
    loss_exp: float = 1.0,
    context_latents: Optional[torch.Tensor] = None,
    state_transition: Optional[StateTransition] = None,
    transition_actions: Optional[torch.Tensor] = None,
    truncate_every: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """L = L_teacher-forcing + L_rollout (equal weights), following upstream app/vjepa_droid/train.py.

    target_latents [B, T, N, D]  stop-gradient targets from the target encoder
    actions        [B, T-1, A]   latent action between frame t and t + 1
    states         [B, T, S]     recorded states
    valid          [B, T] bool   frame validity (padding / episode end); transitions and rollout
                                 steps count only when every frame they depend on is valid
    context_latents              predictor inputs if they differ from the targets (encoder fine-tuning)
    state_transition             use it instead of recorded states inside the rollout branch
                                 to measure deployment-like error
    transition_actions [B, T-1, k, A']  raw control actions per latent step for `state_transition`

    The rollout starts at frame 0 and runs `rollout_steps` steps; `L_rollout` averages the valid
    steps (upstream reduction) and `rollout_final` reports the last step only. `sums` holds the
    masked error sums and valid counts behind these means.
    """
    B, T = target_latents.shape[:2]
    if not 1 <= rollout_steps <= T - 1:
        raise ValueError(f"rollout_steps={rollout_steps} needs at least {rollout_steps + 1} frames, window has {T}")
    target_latents = target_latents.detach()
    inputs = target_latents if context_latents is None else context_latents
    if valid is None:
        valid = torch.ones(B, T, dtype=torch.bool, device=target_latents.device)
    valid = valid.float()

    # Teacher forcing: true history, predict every next latent.
    z_tf = world_model.predict(inputs[:, :-1], actions, states[:, :-1])
    tf_err = latent_l1(z_tf, target_latents[:, 1:], loss_exp)  # [B, T-1]
    tf_mask = valid[:, :-1] * valid[:, 1:]
    tf_loss = (tf_err * tf_mask).sum() / tf_mask.sum().clamp_min(1.0)

    # Rollout: start from the true first latent, feed predictions back.
    z_ar, _ = world_model.rollout(
        inputs[:, :1],
        states[:, :1],
        actions[:, :0],
        actions[:, :rollout_steps],
        state_transition=state_transition,
        future_states=None if state_transition is not None else states[:, 1:rollout_steps],
        transition_actions=None if transition_actions is None else transition_actions[:, :rollout_steps],
        truncate_every=truncate_every,
    )
    ar_err = latent_l1(z_ar, target_latents[:, 1 : rollout_steps + 1], loss_exp)  # [B, R]
    ar_mask = valid[:, : rollout_steps + 1].cumprod(dim=1)[:, 1:]
    rollout_loss = (ar_err * ar_mask).sum() / ar_mask.sum().clamp_min(1.0)
    per_step = (ar_err * ar_mask).sum(0) / ar_mask.sum(0).clamp_min(1.0)

    return {
        "loss": tf_loss + rollout_loss,
        "teacher_forcing": tf_loss,
        "rollout": rollout_loss,
        "rollout_final": per_step[-1],
        "rollout_per_step": per_step.detach(),
        # Masked error sums and valid counts, so metrics can be aggregated across batches exactly.
        "sums": {
            "teacher_forcing": ((tf_err * tf_mask).sum().detach(), tf_mask.sum()),
            "rollout_per_step": ((ar_err * ar_mask).sum(0).detach(), ar_mask.sum(0)),
        },
    }
