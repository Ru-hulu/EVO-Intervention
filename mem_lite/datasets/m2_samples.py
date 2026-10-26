"""M2 的邻居约束扰动与累计前缀标签；所有计算均在原视频帧轴上进行。"""

import random
from bisect import bisect_left
from itertools import pairwise


def sample_event_window(stages, index, total_frames, rng):
    """先均匀采样合法整数长度，再均匀采样该长度的合法起点，无拒绝重试。"""
    stage = stages[index]
    start, end = stage["start_frame"], stage["end_frame"]
    length = end - start
    neighbors = stages[max(0, index - 1):min(len(stages), index + 2)]
    # 整数长度须严格落在 [T-e,T+e]；floor(e) 正好给出两侧整数偏移上限。
    extension = min(item["end_frame"] - item["start_frame"] for item in neighbors) // 5
    sampled_length = rng.randint(length - extension, length + extension)
    required = (4 * length + 4) // 5  # ceil(80% * T)，恰好80%也接受。
    left_limit = (start - (stages[index - 1]["end_frame"] - stages[index - 1]["start_frame"]) // 5
                  if index > 0 else 0)
    right_limit = (end + (stages[index + 1]["end_frame"] - stages[index + 1]["start_frame"]) // 5
                   if index + 1 < len(stages) else total_frames)
    low = max(0, left_limit, start + required - sampled_length)
    high = min(total_frames - sampled_length, right_limit - sampled_length, end - required)
    sampled_start = rng.randint(low, high)
    return [sampled_start, sampled_start + sampled_length]


def build_queue_samples(annotation, kept_frames, timestamps, draws, seed):
    """一个样本保存一组扰动事件；全部前缀共用这些事件，末段只提供标签/邻居长度。"""
    stages = annotation["stages"]
    if len(stages) < 2:
        raise ValueError("M2 至少需要两个连续标注区间，以后一段状态监督前一段事件")
    if any(left["end_frame"] != right["start_frame"] for left, right in pairwise(stages)):
        raise ValueError("M2 标注区间必须按时间顺序连续，不能跨空隙或重叠推断下一段状态")
    if draws < 1:
        raise ValueError("draws_per_video 必须大于0")
    rng = random.Random(seed)
    samples = []
    for draw in range(draws):
        events, windows = [], []
        for index, stage in enumerate(stages[:-1]):
            start, end = sample_event_window(stages, index, annotation["total_frames"], rng)
            indices = set(kept_frames[bisect_left(kept_frames, start):bisect_left(kept_frames, end)])
            indices.update((start, end - 1))  # 本事件自己的扰动首末帧免静止过滤。
            events.append({"frames": [
                {"frame_index": frame, "time_seconds": timestamps[frame], "image": f"frames/{frame:06d}.jpg"}
                for frame in sorted(indices)
            ]})
            windows.append({"stage_id": stage["stage_id"], "event_interval": [stage["start_frame"], stage["end_frame"]],
                            "window": [start, end]})
        samples.append({
            "sample_id": f"queue-{draw:03d}", "inputs": {"events": events},
            "targets": [{"prefix_length": index, "state_description": stage["state_description"]}
                        for index, stage in enumerate(stages[1:], 1)],
            "metadata": {"draw": draw, "windows": windows},
        })
    return samples
