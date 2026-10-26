"""第二阶段：M1 切分监督与事件记忆文本监督，共用冻结的视觉编码器。"""

import torch
from torch import nn

from mem_lite.models.event_encoder import EventEncoder
from mem_lite.models.event_queue import EventQueue
from mem_lite.models.m1_segmenter import M1Segmenter
from mem_lite.models.m2_compressor import M2Compressor
from mem_lite.models.state_decoder import StateTextDecoder


class Stage2Model(nn.Module):
    def __init__(
        self, model_path, m1_config, event_config, m2_config,
        vision_batch_size=16, max_text_sequence_length=2048, dtype=torch.bfloat16,
    ):
        super().__init__()
        self.segmenter = M1Segmenter(model_path, dtype=dtype, **m1_config)
        self.event_encoder = EventEncoder(self.segmenter.visual_feature_dim, **event_config)
        self.compressor = M2Compressor(self.event_encoder.hidden_dim, **m2_config)
        self.decoder = StateTextDecoder(
            model_path, history_dim=self.compressor.hidden_dim, dtype=dtype,
            max_sequence_length=max_text_sequence_length,
        )
        if vision_batch_size < 1:
            raise ValueError("vision_batch_size 必须大于0")
        self.vision_batch_size = vision_batch_size

    def encode_history(self, batch):
        """同一队列中重叠帧只做一次 ViT，同一事件只编码一次，供所有前缀复用。"""
        grids, pixels = batch["image_grid_thw"], batch["pixel_values"]
        offsets = [0, *grids.prod(-1).cumsum(0).tolist()]
        frames = []
        for first in range(0, len(grids), self.vision_batch_size):
            last = min(first + self.vision_batch_size, len(grids))
            frames.extend(self.segmenter.extract_frame_features(pixels[offsets[first]:offsets[last]], grids[first:last]))
        events = self.event_encoder([[frames[index] for index in event] for event in batch["event_frame_indices"]])
        prefixes, offset = [], 0
        for count in batch["queue_event_counts"]:
            # 每个视频/扰动样本独立创建队列，不跨样本、优化步保留事件计算图。
            queue = EventQueue(self.event_encoder.num_event_tokens)
            for event in events[offset:offset + count]:
                queue.append(event)
                prefixes.append(queue.as_tokens())
            offset += count
        return self.compressor(prefixes)

    def memory_loss(self, batch):
        return self.decoder(self.encode_history(batch), batch["state_descriptions"])

    def forward(self, batch):
        # 两条损失之间没有可训练共享参数；M1 不接收文本损失，记忆分支不接收切分损失。
        segmentation = self.segmenter(batch["m1"])
        memory = self.memory_loss(batch["memory"])
        return {
            "m1_loss_sum": segmentation["loss_sum"], "m1_tokens": segmentation["text_tokens"],
            "memory_loss_sum": memory["loss_sum"], "memory_tokens": memory["text_tokens"],
        }

    def parameter_groups(self, m1_learning_rate, memory_learning_rate):
        """分别归一化两个分支的梯度；各参数只能属于一个分支。"""
        return [
            {"name": "m1", "params": [p for p in self.segmenter.parameters() if p.requires_grad],
             "lr": m1_learning_rate},
            {"name": "memory", "params": [p for module in (self.event_encoder, self.compressor, self.decoder)
                                           for p in module.parameters() if p.requires_grad],
             "lr": memory_learning_rate},
        ]
