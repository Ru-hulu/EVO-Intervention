"""与解码和模型无关的 episode 采样、两段交换与标签对齐。"""

import math
import random

from qwen_training.reference_io import FRAME_COUNT


def sample_episode_frames(start, end, rng, fps=30, current=None, min_interval=0.5, max_interval=1.0):
    """帧网格上均匀选当前帧，30fps 下各相邻间隔独立选取 15～30 帧。"""
    if current is None:
        current = rng.randint(start, end)
    low, high = math.ceil(min_interval * fps), math.floor(max_interval * fps)
    if start > end or not start <= current <= end or not 1 <= low <= high:
        raise ValueError("episode 范围或采样间隔无效")
    indices = [current]
    for _ in range(FRAME_COUNT - 1):
        indices.append(indices[-1] - rng.randint(low, high))
    return [index if index >= start else None for index in reversed(indices)]


def make_frame_records(indices, progress, episode_start, episode_end, fps=30):
    """progress 仅含本 episode；末帧暂定为完成点，不用进度四舍五入判断 done。"""
    frames, targets = [], []
    for i, frame in enumerate(indices):
        valid = frame is not None
        frames.append({
            "frame_index": frame, "time_seconds": frame / fps if valid else None,
            "valid": valid, "is_current": i == FRAME_COUNT - 1,
        })
        targets.append({
            "valid": valid,
            "progress": progress[frame - episode_start] if valid else None,
            "done": int(frame == episode_end) if valid else None,
        })
    return frames, targets


def swap_valid_blocks(sample, rng, probability=0.3):
    """只轮转有效位置，帧组与标签共同换位；不修改原样本或空白位置。"""
    frames = sample["frames"]
    slots = [i for i, frame in enumerate(frames) if frame["valid"]]
    order = list(range(len(frames)))
    split = None
    if len(slots) >= 2 and rng.random() < probability:
        split = rng.randint(1, len(slots) - 1)
        for destination, source in zip(slots, slots[split:] + slots[:split]):
            order[destination] = source
    result = dict(sample, frames=[frames[i] for i in order],
                  frame_order=order, block_split=split, order_shuffled=split is not None)
    if "targets" in sample:
        result["targets"] = [sample["targets"][i] for i in order]
    return result


def split_reference_indices(samples, val_fraction=0.1, seed=0):
    """查询原视频整体划分；专家来源视频固定在训练侧，避免验证轨迹作为训练参考。"""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction 必须位于 [0,1)")
    pinned = {(s["reference_source"]["dataset"], s["reference_source"]["video_id"])
              for s in samples if "reference_source" in s}
    by_dataset = {}
    for sample in samples:
        by_dataset.setdefault(sample["dataset"], set()).add(sample["video_id"])
    rng, val_videos = random.Random(seed), set()
    for name, video_ids in sorted(by_dataset.items()):
        candidates = sorted(v for v in video_ids if (name, v) not in pinned)
        rng.shuffle(candidates)
        count = min(int(len(video_ids) * val_fraction + 0.5), len(candidates), len(video_ids) - 1)
        val_videos.update((name, video) for video in candidates[:count])
    train, val = [], []
    for i, sample in enumerate(samples):
        (val if (sample["dataset"], sample["video_id"]) in val_videos else train).append(i)
    return train, val
