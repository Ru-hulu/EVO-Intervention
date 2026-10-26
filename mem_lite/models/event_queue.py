"""单条视频的累计事件队列，向 M2 提供带事件序号的压缩 tokens。"""

import torch

from mem_lite.models.event_encoder import sinusoidal_positions


class EventQueue:
    """按到达顺序保存事件，训练时保留计算图；不保存旧 history 或自动淘汰旧事件。"""

    def __init__(self, num_event_tokens=32):
        self.num_event_tokens = num_event_tokens
        self._events = []

    def __len__(self):
        return len(self._events)

    def append(self, event_tokens):
        """只接收一个已压缩的 [num_event_tokens, hidden_dim] 事件，不 detach。"""
        if event_tokens.ndim != 2 or event_tokens.shape[0] != self.num_event_tokens:
            raise ValueError(f"事件入队需要 [{self.num_event_tokens}, hidden_dim] 的压缩表征")
        self._events.append(event_tokens)

    def as_tokens(self, end=None):
        """读取前 end 个事件，默认全部；返回 [事件数 × num_event_tokens, hidden_dim]。

        事件序号从 0 开始，同一事件所有 tokens 共享一份位置编码。
        仅在读取时加位置编码，重复读取不会往存储表征上累加。
        """
        end = len(self) if end is None else end
        if not 0 < end <= len(self):
            raise ValueError("读取事件队列需要非空且未越界的前缀；新视频暂无事件时不调用 M2")
        events = torch.stack(self._events[:end])
        event_indices = torch.arange(end, device=events.device)
        positions = sinusoidal_positions(event_indices, events.shape[-1], events.dtype)
        return (events + positions[:, None, :]).flatten(0, 1)

    def clear(self):
        """新视频或新的训练前向开始时清空，释放队列持有的计算图引用。"""
        self._events.clear()
