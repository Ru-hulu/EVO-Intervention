"""StreamForest 的 ToMe 加权合并，移除未使用的逆变换。来源见 README。"""
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import functional as F


def matching_scores(tokens, num_heads=1):
    metric = tokens.unflatten(-1, (num_heads, -1)).mean(-2).float()
    metric = F.normalize(metric, dim=-1)
    with torch.autocast(tokens.device.type, enabled=False):
        return metric[..., ::2, :] @ metric[..., 1::2, :].transpose(-1, -2)


@torch.no_grad()
def merge_tokens(tokens, target, num_heads=1):
    """[N, D] → [min(N, target), D]；按累计 token 权重合并，不补造 tokens。"""
    size = tokens.new_ones((len(tokens), 1), dtype=torch.float32)
    while len(tokens) > target:
        count = min(len(tokens) - target, len(tokens) // 2)
        scores = matching_scores(tokens, num_heads)
        values, destinations = scores.max(-1)
        order = values.argsort(descending=True)
        selected, kept = order[:count], order[count:]
        destination = destinations[selected]

        def merge(values, selected, kept, destination):
            source, dest = values[::2], values[1::2]
            dest = dest.scatter_add(0, destination[:, None].expand(-1, values.shape[-1]), source[selected])
            return torch.cat((source[kept], dest))

        weighted = merge(tokens.float() * size, selected, kept, destination)
        size = merge(size, selected, kept, destination)
        tokens = (weighted / size).to(tokens.dtype)
    return tokens
