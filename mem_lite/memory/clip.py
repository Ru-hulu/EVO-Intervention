"""Frozen CLIP CLS descriptors, used only to choose Forest event boundaries."""

import hashlib
import json

import torch

from .vision import _vision_sha256


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ClipBoundaryEncoder:
    """One main-view image -> final CLS after layer norm, without projection.

    FP32/eager and the explicit PIL processor avoid outer autocast/processor
    backend changes between offline cache creation and online inference.
    Loading is local-only; the caller supplies a downloaded, pinned model.
    """

    def __init__(self, model_path, device):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.processor = CLIPImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model, loading = CLIPVisionModel.from_pretrained(
            model_path, local_files_only=True, dtype=torch.float32, attn_implementation="eager",
            output_loading_info=True,
        )
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError("CLIP 视觉权重不完整或与配置不匹配，不使用随机初始化的边界编码器")
        self.model.requires_grad_(False).eval()
        config = self.model.config.to_dict()
        for key in ("_name_or_path", "transformers_version", "_commit_hash"):
            config.pop(key, None)
        self.provenance = {
            "weights_sha256": _vision_sha256(self.model.state_dict()),
            "config_sha256": _json_hash(config),
            "processor_sha256": _json_hash(self.processor.to_dict()),
            "feature": "last_cls_post_layernorm_no_projection",
            "dtype": "float32", "attention": "eager",
        }
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, image):
        self.model.eval()
        device = next(self.model.parameters()).device
        pixels = self.processor(images=[image.convert("RGB")], return_tensors="pt")["pixel_values"]
        with torch.autocast(device.type, enabled=False):
            return self.model(pixel_values=pixels.to(device=device, dtype=torch.float32)).pooler_output[0]
