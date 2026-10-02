"""三视角单帧编码：共享 Qwen 视觉编码器，按顶部、左侧、右侧拼接 token。"""

import torch
from torch.nn.utils.rnn import pad_sequence


def encode_three_views(visual, image_processor, images):
    """images 为批量的 [顶部图像, 左侧图像, 右侧图像]。

    每张图像为 RGB PIL 图像或 uint8 CHW tensor（0–255）。
    也可直接传入 [B,3,3,H,W] 的 uint8 tensor。
    返回 visual_features [B,N,D] 与 visual_mask [B,N]，保留各视角全部输出 token。
    """
    if any(len(views) != 3 for views in images):
        raise ValueError("每个样本需要按顶部、左侧、右侧顺序提供三张图像")

    flat_images = [image for views in images for image in views]
    inputs = image_processor(images=flat_images, return_tensors="pt")
    grid = inputs["image_grid_thw"].to(visual.device)
    # 按独立图片处理，原生视觉注意力不会混合不同视角或不同样本。
    tokens = visual(
        inputs["pixel_values"].to(device=visual.device, dtype=visual.dtype),
        grid_thw=grid, return_dict=True,
    ).pooler_output

    # 原生 Merger 输出维度为语言隐藏维度；这里只拼接，不额外池化或压缩。
    counts = grid.prod(dim=-1) // visual.spatial_merge_size**2
    lengths = counts.reshape(-1, 3).sum(dim=-1) # 三个视角作为一行，求和得到一帧的三个视角有多少个token
    features = pad_sequence(tokens.split(lengths.tolist()), batch_first=True) # B * N * token_dim -> # B * Nmax * token_dim 三个视角都在N里
    mask = torch.arange(features.shape[1], device=features.device)[None, :] < lengths[:, None]
    return {"visual_features": features, "visual_mask": mask}

