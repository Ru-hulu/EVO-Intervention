"""Frozen Qwen3.5 vision tower returning merged spatial tokens for JEPA."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


class Qwen3_5VisionEncoder(torch.nn.Module):
    def __init__(self, model_path: str | Path, img_size: tuple[int, int]):
        super().__init__()
        root = Path(model_path).expanduser()
        config = AutoConfig.from_pretrained(root, local_files_only=True).vision_config
        config._attn_implementation = "sdpa"
        self.visual = Qwen3_5VisionModel(config)
        self.image_processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
        self.img_height, self.img_width = img_size
        self.patch_size = config.patch_size * config.spatial_merge_size
        self.tubelet_size = config.temporal_patch_size
        self.embed_dim = config.out_hidden_size
        self.image_mean = self.image_processor.image_mean
        self.image_std = self.image_processor.image_std
        if any(size % self.patch_size for size in img_size):
            raise ValueError(f"Qwen image size {img_size} must be divisible by {self.patch_size}")

        # Load only model.visual.*; language weights are never instantiated or read.
        index = json.loads((root / "model.safetensors.index.json").read_text())
        prefix = "model.visual."
        weight_map = {key: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
        if not weight_map:
            raise ValueError(f"No {prefix} weights in {root}")
        state = {}
        for shard in sorted(set(weight_map.values())):
            with safe_open(root / shard, framework="pt", device="cpu") as checkpoint:
                for key, filename in weight_map.items():
                    if filename == shard:
                        state[key.removeprefix(prefix)] = checkpoint.get_tensor(key)
        self.visual.load_state_dict(state, strict=True, assign=True)
        self.loaded_tensors = len(state)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Repeated static-frame clips [B, 3, tubelet, H, W] -> [B, tokens_per_frame, embed_dim]."""
        images = clips[:, :, 0]
        inputs = self.image_processor(
            images=images,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,  # world_model.preprocess already applies Qwen's mean/std
            device=images.device,
            return_tensors="pt",
        )
        outputs = self.visual(
            hidden_states=inputs.pixel_values.to(device=images.device, dtype=self.visual.dtype),
            grid_thw=inputs.image_grid_thw.to(images.device),
            return_dict=True,
        )
        # The native merger produces row-major spatial tokens; no language model is involved.
        return outputs.pooler_output.view(images.shape[0], -1, self.embed_dim)
