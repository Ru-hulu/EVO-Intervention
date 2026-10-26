"""把单个事件的变长视觉序列压缩为固定数量的事件 tokens。"""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


def sinusoidal_positions(indices, width, dtype):
    """无固定长度上限的正余弦位置编码；返回 [*indices.shape, width]。"""
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=indices.device, dtype=torch.float32) * (-math.log(10000.0) / width)
    )
    angles = indices.float().unsqueeze(-1) * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :width].to(dtype)


class EventCompressionBlock(nn.Module):
    """Pre-norm 交叉注意力和 FFN，两个子层分别带残差连接。"""

    def __init__(self, hidden_dim, num_heads, ffn_multiplier, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, queries, visual_tokens, padding_mask):
        visual = self.visual_norm(visual_tokens)
        update, _ = self.attention(
            self.query_norm(queries), visual, visual, key_padding_mask=padding_mask, need_weights=False,
        )
        queries = queries + self.residual_dropout(update)
        return queries + self.residual_dropout(self.ffn(self.ffn_norm(queries)))


class EventEncoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=512, num_event_tokens=32, num_layers=4,
        num_heads=8, ffn_multiplier=4, dropout=0.0,
    ):
        super().__init__()
        if num_event_tokens < 1 or num_layers < 1:
            raise ValueError("事件 token 数和编码器层数必须为正数")
        self.hidden_dim = hidden_dim
        self.num_event_tokens = num_event_tokens
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(num_event_tokens, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([
            EventCompressionBlock(hidden_dim, num_heads, ffn_multiplier, dropout) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, events):
        """events[b][f] 是第 b 个事件第 f 帧的 [空间 token 数, input_dim] 表征。

        帧数和每帧 token 数均可变化；只接收视觉表征，不接任务指令、专家或语言隐藏状态。
        返回 [事件批大小, num_event_tokens, hidden_dim]，不在此添加事件序号位置编码。
        """
        if not events:
            raise ValueError("至少需要一个非空事件")
        sequences, frame_positions = [], []
        for frames in events:
            if not frames or any(frame.shape[0] == 0 for frame in frames):
                raise ValueError("事件及其帧表征不能为空，不能对全 padding 序列进行交叉注意力")
            sequences.append(torch.cat(frames, dim=0))
            # 同一帧的空间 tokens 共用帧位置；每个事件从帧序号 0 重新计数。
            frame_positions.append(torch.cat([
                torch.full((len(frame),), index, device=frame.device, dtype=torch.long)
                for index, frame in enumerate(frames)
            ]))
        padded = pad_sequence(sequences, batch_first=True)
        positions = pad_sequence(frame_positions, batch_first=True)
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=padded.device)
        padding_mask = torch.arange(padded.shape[1], device=padded.device)[None, :] >= lengths[:, None]
        visual = self.input_projection(padded.to(self.input_projection.weight.dtype))
        visual = visual + sinusoidal_positions(positions, self.hidden_dim, visual.dtype)
        queries = self.queries.to(visual.dtype).unsqueeze(0).expand(len(events), -1, -1)
        for layer in self.layers:
            queries = layer(queries, visual, padding_mask)
        return self.output_norm(queries)


def encode_event_intervals(encoder, frame_features, intervals):
    """按观测队列的零基 [start,end) 边界提取事件；不修改、消费或重新抽样原观测队列。"""
    events = []
    for start, end in intervals:
        if not 0 <= start < end <= len(frame_features):
            raise ValueError(f"事件区间 [{start},{end}) 超出含 {len(frame_features)} 帧的观测队列")
        events.append(frame_features[start:end])
    return encoder(events)
