"""训练缓存与在线预测共用的逐帧记忆更新。"""

from copy import deepcopy

import torch

from .manager import MemoryManager
from .splitting import DEPTH_SPEC, override_memory_config, split_method, validate_history_source


class StreamMemory:
    def __init__(self, encoder, processor, memory_config, max_pixels=65536, min_pixels=4096,
                 *, clip_model_path=None):
        self.encoder, self.processor = encoder, processor
        self.memory_config = override_memory_config(memory_config)
        manager_config = dict(self.memory_config)
        configured_clip_path = manager_config.pop("clip_model_path", None)
        clip_path = clip_model_path or configured_clip_path
        self.memory = MemoryManager(encoder.hidden_size, **manager_config)
        self.boundary_encoder = None
        if split_method(memory_config) == "max_depth":
            if not clip_path:
                raise ValueError("max_depth 需要 --clip-model-path 或 memory_config.clip_model_path")
            from .clip import ClipBoundaryEncoder
            self.boundary_encoder = ClipBoundaryEncoder(clip_path, next(encoder.parameters()).device)
        elif clip_model_path is not None:
            raise ValueError("clip_model_path 仅用于 max_depth")
        self.image_options = {"max_pixels": max_pixels, "min_pixels": min_pixels}
        self.reset()

    def bind_history_source(self, source, *, require_provenance=False):
        """Attach real encoder hashes on build; verify them on deployment."""
        source = deepcopy(source)
        config = source["memory_config"]
        if config != self.memory_config:
            raise ValueError("history_source.memory_config 与实际记忆流配置不一致")
        if (source["max_pixels"] != self.image_options["max_pixels"]
                or source["min_pixels"] != self.image_options["min_pixels"]):
            raise ValueError("history_source 图像预处理与实际记忆流不一致")
        if require_provenance:
            validate_history_source(source)
        if self.boundary_encoder is not None:
            actual = {**DEPTH_SPEC, "clip": self.boundary_encoder.provenance}
            if "event_split" in source and source["event_split"] != actual:
                raise ValueError("CLIP 权重、预处理或切分规则与缓存/检查点不一致")
            source["event_split"] = deepcopy(actual)
        validate_history_source(source)
        return source

    def reset(self):
        self.memory.reset()
        self.session = None
        self.last_frame_index = None

    @torch.no_grad()
    def step(self, image, frame_index, episode_id, task_prompt):
        """返回当前帧之前的历史，随后将当前主视角写入记忆。"""
        session = (episode_id, task_prompt)
        if session != self.session:
            self.reset()
            self.session = session
        if self.last_frame_index is not None and frame_index <= self.last_frame_index:
            raise ValueError("同一视频的帧号必须递增；重播请先 reset()")
        history = self.memory.get_memory_tokens()
        inputs = self.processor(images=[image], return_tensors="pt", **self.image_options)
        frame = self.encoder(inputs["pixel_values"], inputs["image_grid_thw"], [frame_index])[0]
        descriptor = self.boundary_encoder(image) if self.boundary_encoder is not None else None
        self.memory.update(frame, boundary_descriptor=descriptor)
        self.last_frame_index = frame_index
        return history
