"""M1 四类扰动、关键帧队列索引和自回归 JSON 标签。无模型或解码器依赖。"""

import json
import random
from bisect import bisect_left
from pathlib import Path

# (起点偏移方向, 终点偏移方向, 左扰动百分比上限, 右扰动百分比上限)
CASE_RULES = {
    "case1": (1, 1, 40, 10),
    "case2": (-1, 1, 20, 20),
    "case3": (-1, -1, 10, 40),
    "case4": (1, -1, 20, 20),
}


def read_annotation(path):
    annotation = json.loads(Path(path).read_text(encoding="utf-8"))
    if annotation["frame_index_base"] != 0 or annotation["frame_range_convention"] != "[start_frame, end_frame)":
        raise ValueError("标注需采用 0 起始帧号和 [start_frame, end_frame) 区间")
    for stage in annotation["stages"]:
        if not 0 <= stage["start_frame"] < stage["end_frame"] <= annotation["total_frames"]:
            raise ValueError(f"阶段区间越界或为空：{stage}")
    return annotation


def make_window(start, end, total_frames, case, left_percent, right_percent):
    """百分比按原事件时长计算，整数帧采用四舍五入，正样本严格大于 80%。"""
    length = end - start
    left = (length * left_percent + 50) // 100
    right = (length * right_percent + 50) // 100
    left_sign, right_sign, _, _ = CASE_RULES[case]
    window = [max(0, start + left_sign * left), min(total_frames, end + right_sign * right)]
    intersection = [max(start, window[0]), min(end, window[1])]
    overlap = max(0, intersection[1] - intersection[0])
    valid = 5 * overlap > 4 * length  # 避免浮点数将恰好 80% 判为有效。
    return {
        "window": window,
        "target_interval": intersection if valid else None,
        "valid": int(valid),
        "coverage": overlap / length,
        "offset_percent": [left_percent, right_percent],
        "offset_frames": [left, right],
    }


def sample_windows(annotation, config):
    rng = random.Random(config["seed"])
    windows = []
    for stage in annotation["stages"]:
        start, end = stage["start_frame"], stage["end_frame"]
        for case, count in config["samples_per_case"].items():
            _, _, left_max, right_max = CASE_RULES[case]
            for draw in range(count):
                sample = make_window(
                    start,
                    end,
                    annotation["total_frames"],
                    case,
                    rng.randrange(0, left_max + 1, 2),
                    rng.randrange(0, right_max + 1, 2),
                )
                sample.update(stage_id=stage["stage_id"], case=case, draw=draw, event_interval=[start, end])
                windows.append(sample)
    return windows


def build_sample(window, kept_frames, timestamps, task_prompt, expert_tokens_path=None):
    """公共观测切片 + 本样本专属目标边界；返回 inputs/target/metadata，空队列返回 None。"""
    start, end = window["window"]
    indices = set(kept_frames[bisect_left(kept_frames, start) : bisect_left(kept_frames, end)])
    target_interval = window["target_interval"]
    if target_interval is not None:
        indices.update((target_interval[0], target_interval[1] - 1))
    indices = sorted(indices)
    if not indices:
        return None
    target = {"valid": window["valid"]}
    if target_interval is not None:
        target.update(start_index=indices.index(target_interval[0]), end_index=indices.index(target_interval[1] - 1) + 1)
    sample_id = f"stage-{window['stage_id']:03d}-{window['case']}-{window['draw']:03d}"
    return {
        "sample_id": sample_id,
        "inputs": {
            "task_prompt": task_prompt,
            "expert_tokens_path": expert_tokens_path,
            "frames": [
                {
                    "queue_index": i,
                    "frame_index": frame,
                    "time_seconds": timestamps[frame],
                    "image": f"frames/{frame:06d}.jpg",
                }
                for i, frame in enumerate(indices)
            ],
        },
        "target": target,
        "target_text": json.dumps(target, ensure_ascii=False),
        "metadata": window,
    }


def format_m1_prompt(sample):
    """只将任务、队列编号和实际时间交给模型；不包含 case、真值边界或状态描述。"""
    frames = sample["inputs"]["frames"]
    positions = ", ".join(f"{f['queue_index']}@{f['time_seconds'] - frames[0]['time_seconds']:.3f}s" for f in frames)
    return (
        f"任务：{sample['inputs']['task_prompt']}\n"
        f"观测按队列顺序输入，编号与相对时间为：{positions}。\n"
        "根据上述观测判断是否存在适合切分的事件片段。只输出 JSON，不作解释。"
        '无效时输出 {"valid": 0}；有效时输出 {"valid": 1, "start_index": 起点序号, '
        '"end_index": 终点序号}。区间左闭右开，end_index 不包含在片段内，可以等于队列长度。'
    )


def read_training_sample(sample, sample_dir):
    """按队列顺序读取 PIL 图像；专家路径仅保留元数据，不加载或伪造专家 tokens。"""
    from PIL import Image

    images = []
    for frame in sample["inputs"]["frames"]:
        with Image.open(Path(sample_dir) / frame["image"]) as image:
            images.append(image.convert("RGB"))
    return {
        "images": images,
        "prompt": format_m1_prompt(sample),
        "expert_tokens_path": sample["inputs"]["expert_tokens_path"],
        "response": sample["target_text"],
    }
