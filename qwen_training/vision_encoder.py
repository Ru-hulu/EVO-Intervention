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


def encode_history(visual, image_processor, images, frame_mask=None):
    """直接编码传入的三视角全部历史帧，通过共享视觉编码器进行 MEM 时序融合。

    输入参数：
        visual：已加载的 Qwen 视觉编码器，三个视角共享它的权重。
        image_processor：与该编码器匹配的图像预处理器，将图片转为 pixel_values 和网格信息。
        images：按 [B,3,L] 组织的图片列表，B 是样本数，3 个视角依次为顶部、左侧、右侧。
            L 是每个视角输入的总帧数，包含历史帧、最后的当前帧以及可能的占位帧。
            每张图片为 RGB PIL 或 uint8 [3,H,W] tensor，像素范围为 0–255。
            同尺寸时也可直接传 uint8 [B,3,L,3,H,W]，后一个 3 表示 RGB 通道数。
        frame_mask：可选有效帧标记，形状 [B,3,L]；True 表示真实帧，False 表示占位帧。
            默认 None，认为传入的帧全部有效；每个视角的当前帧必须有效。

    帧选择和历史不足时的占位由输入端准备；各视角长度 L 相同，时间从旧到新。
    编码器按输入帧顺序编码，当前帧位置为 0，MEM 使用的帧数 K 就是 L。
    MEM 时序注意力固定在第 4、8、12……层执行。

    返回 visual_features [B,N,D] / visual_mask [B,N]，按视角、时间拼接所有输入帧。
    N 是批内补齐后的 token 数，D 是 Merger 输出维度；此步尚未筛选当前帧。
    无效帧 token 和样本间补齐 token 均为 0，对应 visual_mask=False。
    """
    batch_size = len(images)  # 样本数 B。
    num_frames = len(images[0][0])  # 每个视角传入多少帧，就编码多少帧：K=L。
    if num_frames < 1 or any(len(views) != 3 or any(len(stream) != num_frames for stream in views)
                            for views in images):
        raise ValueError("images 需要按 [B,3,L] 排列，且 L >= 1")

    shape = (batch_size, 3, num_frames)  # [B,3,L]。
    valid = torch.ones(shape, dtype=torch.bool) if frame_mask is None else torch.as_tensor(frame_mask).bool()
    if valid.shape != shape or not valid[..., -1].all():
        raise ValueError("frame_mask 需为 [B,3,L]，且三个视角的当前帧必须有效")

    flat_images = []
    for views in images:  # 取一个样本的三个视角。
        for stream in views:  # 取一个视角的全部帧，按从旧到新的顺序加入列表。
            flat_images.extend(stream) # [第1个视角的第一帧到最后一帧 第2个视角的第一帧到最后一帧 第3个视角的第一帧到最后一帧] B × 3 × L 张图片的列表
    inputs = image_processor(images=flat_images, return_tensors="pt")
    grid = inputs["image_grid_thw"].to(visual.device)  # [B*3*L,3]，每行 [T=1,H,W]。
    tokens = mem_vision_forward(
        visual, inputs["pixel_values"], grid, num_frames, frame_mask=valid.reshape(-1, num_frames),
    )  # [所有帧的 Merger token 总数,D]。
    counts = grid.prod(dim=-1) // visual.spatial_merge_size**2  # [B*3*L]。
    lengths = counts.reshape(batch_size, -1).sum(dim=-1).tolist()  # 每个样本的 token 数。
    token_mask = valid.to(visual.device).flatten().repeat_interleave(counts)  # 与 tokens 一一对应。
    features = pad_sequence(tokens.split(lengths), batch_first=True)  # [B,Nmax,D]。
    mask = pad_sequence(token_mask.split(lengths), batch_first=True, padding_value=False)  # [B,Nmax]。
    return {"visual_features": features, "visual_mask": mask}
