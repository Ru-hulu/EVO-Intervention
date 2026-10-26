"""从第二阶段检查点加载冻结事件记忆，仅用于第三阶段的历史缓存构造。"""

import torch
from torch import nn
from peft import set_peft_model_state_dict

from mem_lite.models.event_encoder import EventEncoder
from mem_lite.models.event_queue import EventQueue
from mem_lite.models.m1_segmenter import M1Segmenter
from mem_lite.models.m2_compressor import M2Compressor


class FrozenEventMemory(nn.Module):
    def __init__(self, model_path, checkpoint_path, vision_batch_size=16, dtype=torch.bfloat16):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "mem-lite-stage2-v1":
            raise ValueError("历史缓存必须来自第二阶段检查点，不能使用随机事件编码器/M2")
        config, state = checkpoint["config"], checkpoint["model"]
        m1_config = {**config["m1"], "gradient_checkpointing": False}
        self.segmenter = M1Segmenter(model_path, dtype=dtype, **m1_config)
        set_peft_model_state_dict(self.segmenter.qwen, state["m1_lora"])
        self.event_encoder = EventEncoder(self.segmenter.visual_feature_dim, **config["event_encoder"])
        self.compressor = M2Compressor(self.event_encoder.hidden_dim, **config["m2_compressor"])
        self.event_encoder.load_state_dict(state["event_encoder"])
        self.compressor.load_state_dict(state["m2_compressor"])
        if vision_batch_size < 1:
            raise ValueError("vision_batch_size 必须大于0")
        self.vision_batch_size = vision_batch_size
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # 调用方的 train() 也不能打开冻结记忆模块的 dropout。
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch):
        """一条视频的有序事件帧索引 → 每个累计前缀的128个历史tokens。"""
        grids, pixels = batch["image_grid_thw"], batch["pixel_values"]
        offsets = [0, *grids.prod(-1).cumsum(0).tolist()]
        frames = []
        for first in range(0, len(grids), self.vision_batch_size):
            last = min(first + self.vision_batch_size, len(grids))
            frames.extend(self.segmenter.extract_frame_features(pixels[offsets[first]:offsets[last]], grids[first:last]))
        queue, histories = EventQueue(self.event_encoder.num_event_tokens), []
        for indices in batch["event_frame_indices"]:
            event = self.event_encoder([[frames[index] for index in indices]])[0]
            queue.append(event)
            histories.append(self.compressor([queue.as_tokens()])[0].float().cpu())
        if not histories:
            return torch.empty(0, 128, self.compressor.hidden_dim)
        return torch.stack(histories)
