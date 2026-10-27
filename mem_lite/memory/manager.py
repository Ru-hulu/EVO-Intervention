"""StreamForest 短期窗口与事件记忆森林，适配可变大小的 Qwen 视觉网格。"""
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch.nn import functional as F

from .tome import matching_scores, merge_tokens


@dataclass
class Event:
    tokens: torch.Tensor
    time: float
    merges: int = 1


class MemoryManager:
    """无可训练参数；每个实例管理一条按时间排序、单视角的视觉流。"""

    def __init__(
        self, hidden_size, num_attention_heads=1, st_memory_windows=(1, 6),
        st_memory_tokens=(64, 16), event_split_window=4,
        long_memory_tokens_per_frame=16, long_memory_tokens_quota=800,
        sim_weight_g=0.4, time_weight_a=0.2, merge_weight_b=0.4,
    ):
        # 数量必须为正，避免无法推进的窗口或 token 合并循环。
        if min(*st_memory_windows, *st_memory_tokens, event_split_window,
               long_memory_tokens_per_frame, long_memory_tokens_quota) < 1:
            raise ValueError("记忆窗口和 token 预算必须为正数")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.st_memory_windows = st_memory_windows
        self.st_memory_tokens = st_memory_tokens
        self.event_split_window = event_split_window
        self.long_memory_tokens_per_frame = long_memory_tokens_per_frame
        self.long_memory_tokens_quota = long_memory_tokens_quota
        self.weights = (sim_weight_g, time_weight_a, merge_weight_b)
        self.reset()

    @property
    def token_budget(self):
        """包括当前、短期缓冲和长期记忆的总上限。"""
        now, short = self.st_memory_windows
        return (now * self.st_memory_tokens[0]
                + (short + self.event_split_window - 1) * min(self.st_memory_tokens)
                + self.long_memory_tokens_quota)

    def reset(self):
        self.memory_now, self.memory_short, self.memory_long = [], [], []
        self.frame_similarities, self.event_similarities = [], []
        self.last_frame = None
        self.frames_seen = self.frames_in_long = 0

    def _merge(self, tokens, target):
        return merge_tokens(tokens, target, self.num_attention_heads)

    def _similarity(self, left, right):
        tokens = torch.cat((left.tokens, right.tokens))
        count = len(tokens) // 2
        scores = matching_scores(tokens, self.num_attention_heads).max(-1).values
        return scores.topk(count).values.mean().item()

    @torch.no_grad()
    def update(self, frame):
        """接收一帧 [N, hidden_size] 特征，并立即维护预算。"""
        # 池化仅用于事件边界相似度；完整空间 tokens 仍参与记忆合并。
        descriptor = frame.float().mean(0)
        if self.last_frame is not None:
            self.frame_similarities.append(F.cosine_similarity(self.last_frame, descriptor, dim=0).item())
        self.last_frame = descriptor
        self.frames_seen += 1
        self.memory_now.append(self._merge(frame.detach(), self.st_memory_tokens[0]))
        if len(self.memory_now) > self.st_memory_windows[0]:
            self.memory_short.append(self._merge(self.memory_now.pop(0), self.st_memory_tokens[1]))
        self._split_events()
        self._compress_long_memory()

    def _split_events(self):
        while len(self.memory_short) >= self.st_memory_windows[1] + self.event_split_window:
            window = self.frame_similarities[:self.event_split_window]
            count = min(range(len(window)), key=window.__getitem__) + 1
            tokens = self._merge(torch.cat(self.memory_short[:count]), count * self.long_memory_tokens_per_frame)
            event = Event(tokens, self.frames_in_long + (count - 1) / 2)
            if self.memory_long:
                self.event_similarities.append(self._similarity(self.memory_long[-1], event))
            self.memory_long.append(event)
            del self.memory_short[:count]
            del self.frame_similarities[:count]
            self.frames_in_long += count

    def _compress_long_memory(self):
        total = sum(len(event.tokens) for event in self.memory_long)
        while total > self.long_memory_tokens_quota and len(self.memory_long) > 1:
            pairs = list(pairwise(self.memory_long))
            merge_counts = [(a.merges + b.merges) / 2 for a, b in pairs]
            g, a, b = self.weights
            penalties = [
                g * (1 - similarity) + a * (left.time + right.time) / (2 * (self.frames_seen - 1))
                + b * count / max(merge_counts)
                for (left, right), count, similarity in zip(pairs, merge_counts, self.event_similarities, strict=True)
            ]
            index = min(range(len(penalties)), key=penalties.__getitem__)
            left, right = pairs[index]
            size = len(left.tokens) + len(right.tokens)
            # 原实现以固定帧网格折半；这里按实际 token 数折半，支持不同图像尺寸。
            tokens = self._merge(torch.cat((left.tokens, right.tokens)), math.ceil(size / 2))
            time = (left.time * len(left.tokens) + right.time * len(right.tokens)) / size
            self.memory_long[index:index + 2] = [Event(tokens, time, left.merges + right.merges)]
            total -= size - len(tokens)
            del self.event_similarities[index]
            for neighbor in (index - 1, index):
                if 0 <= neighbor < len(self.memory_long) - 1:
                    self.event_similarities[neighbor] = self._similarity(
                        self.memory_long[neighbor], self.memory_long[neighbor + 1],
                    )
        # 很小的预算也要生效，不能因为只剩一个事件而超额。
        if total > self.long_memory_tokens_quota:
            self.memory_long[0].tokens = self._merge(self.memory_long[0].tokens, self.long_memory_tokens_quota)

    def get_memory_tokens(self):
        """按长期→短期→当前顺序返回 [N, D]；没有观测时返回 None。"""
        parts = [event.tokens for event in self.memory_long] + self.memory_short + self.memory_now
        return torch.cat(parts) if parts else None
