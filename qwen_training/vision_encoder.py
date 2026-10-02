"""共享 Qwen 视觉编码器的三视角单帧 / 历史帧入口。"""

import torch
from torch.nn.utils.rnn import pad_sequence

from qwen_training.mem_attention import mem_vision_forward


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


def encode_history(visual, image_processor, images, num_frames=None, frame_stride=1,
                   frame_interval=1.0, frame_times=None, frame_mask=None, temporal_layers=None):
    """images 按 [B,3,L] 组织，三个视角依次为顶部、左侧、右侧，时间从旧到新。

    每张图片格式与 encode_three_views 相同，也可传 uint8 [B,3,L,3,H,W]。
    从最新帧向前每隔 frame_stride 个输入位置取一帧，保留 num_frames=K 帧；
    K 默认为全部采样结果，不足 K 时在左侧补齐。所有流的输入长度 L 需相同，
    缺失历史可用占位图片配合 frame_mask [B,3,L]（True=有效，末帧必须有效）。
    frame_interval 是输入相邻帧的秒数；也可用 frame_times [B,3,L] 提供实际时间戳。
    返回 visual_features [B,N,D] / visual_mask [B,N]，按视角、时间拼接所有帧。
    此步尚未筛选当前帧；无效帧 token 和样本间补齐 token 均为 0，mask=False。
    """
    batch_size, source_frames = len(images), len(images[0][0])
    if source_frames < 1 or any(len(views) != 3 or any(len(stream) != source_frames for stream in views)
                                for views in images):
        raise ValueError("images 需要按 [B,3,L] 排列，且 L >= 1")
    if frame_stride < 1 or frame_interval <= 0 or (num_frames is not None and num_frames < 1):
        raise ValueError("num_frames、frame_stride 和 frame_interval 必须为正数")

    indices = list(range(source_frames - 1, -1, -frame_stride))  # 从当前帧向过去取样。
    num_frames = len(indices) if num_frames is None else num_frames
    indices = indices[:num_frames][::-1]  # 长度 <= K，恢复从旧到新的顺序。
    padding = num_frames - len(indices)
    indices = [indices[0]] * padding + indices  # 左侧复制最早的采样帧，仅作形状占位。

    shape = (batch_size, 3, source_frames)
    valid = torch.ones(shape, dtype=torch.bool) if frame_mask is None else torch.as_tensor(frame_mask).bool()
    # 时间戳保留 FP64 到减去当前时刻之后，支持绝对秒数。
    times = (torch.arange(source_frames, dtype=torch.float64).expand(shape) * frame_interval
             if frame_times is None else torch.as_tensor(frame_times, dtype=torch.float64))
    if valid.shape != shape or times.shape != shape or not valid[..., -1].all():
        raise ValueError("frame_mask / frame_times 需为 [B,3,L]，且三个视角的当前帧必须有效")
    valid = valid[..., indices].clone()  # [B,3,L] → [B,3,K]。
    valid[..., :padding] = False
    times = times[..., indices]  # [B,3,K]，保留真实采样间隔。

    flat_images = [stream[i] for views in images for stream in views for i in indices]
    inputs = image_processor(images=flat_images, return_tensors="pt")
    grid = inputs["image_grid_thw"].to(visual.device)  # [B*3*K,3]，每行 [T=1,H,W]。
    tokens = mem_vision_forward(
        visual, inputs["pixel_values"], grid, num_frames, temporal_layers,
        frame_times=times.reshape(-1, num_frames), frame_mask=valid.reshape(-1, num_frames),
    )  # [所有帧的 Merger token 总数,D]。
    counts = grid.prod(dim=-1) // visual.spatial_merge_size**2  # [B*3*K]。
    lengths = counts.reshape(batch_size, -1).sum(dim=-1).tolist()  # 每个样本的 token 数。
    token_mask = valid.to(visual.device).flatten().repeat_interleave(counts)  # 与 tokens 一一对应。
    features = pad_sequence(tokens.split(lengths), batch_first=True)  # [B,Nmax,D]。
    mask = pad_sequence(token_mask.split(lengths), batch_first=True, padding_value=False)  # [B,Nmax]。
    return {"visual_features": features, "visual_mask": mask}
