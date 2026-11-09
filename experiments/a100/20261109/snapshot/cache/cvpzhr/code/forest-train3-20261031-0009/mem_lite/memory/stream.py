"""训练缓存与在线预测共用的逐帧记忆更新。"""

import torch

from .manager import MemoryManager


class StreamMemory:
    def __init__(self, encoder, processor, memory_config, max_pixels=65536, min_pixels=4096):
        self.encoder, self.processor = encoder, processor
        self.memory = MemoryManager(encoder.hidden_size, **memory_config)
        self.image_options = {"max_pixels": max_pixels, "min_pixels": min_pixels}
        self.reset()

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
        self.memory.update(frame)
        self.last_frame_index = frame_index
        return history
