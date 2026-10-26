"""M2：用固定128个 query 将累计事件队列压缩为历史 tokens。"""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from mem_lite.models.event_encoder import EventCompressionBlock


class M2Compressor(nn.Module):
    """四层交叉注意力与 FFN；每次独立读取全部事件，不接收或缓存旧 history。"""

    def __init__(self, input_dim=512, hidden_dim=512, num_heads=8, ffn_multiplier=4, dropout=0.0):
        super().__init__()
        self.num_history_tokens = 128
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(self.num_history_tokens, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        # 只复用模块实现，与事件编码器以及其余层均不共享参数。
        self.layers = nn.ModuleList([
            EventCompressionBlock(hidden_dim, num_heads, ffn_multiplier, dropout) for _ in range(4)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, event_queues):
        """event_queues 是张量列表，各项为 [累计事件数×32, input_dim]。

        输入来自 EventQueue.as_tokens()，已经带有事件序号，不在此重复加位置编码。
        不同视频或队列前缀可以组成变长批次；返回 [批大小, 128, hidden_dim]。
        """
        if not event_queues or any(len(tokens) == 0 for tokens in event_queues):
            raise ValueError("M2 需要至少一条非空事件队列，空历史不构造伪造事件")
        padded = pad_sequence(event_queues, batch_first=True)
        lengths = torch.tensor([len(tokens) for tokens in event_queues], device=padded.device)
        padding_mask = torch.arange(padded.shape[1], device=padded.device)[None, :] >= lengths[:, None]
        events = self.input_projection(padded.to(self.input_projection.weight.dtype))
        queries = self.queries.to(events.dtype).unsqueeze(0).expand(len(event_queues), -1, -1)
        for layer in self.layers:
            queries = layer(queries, events, padding_mask)
        return self.output_norm(queries)
