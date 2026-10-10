"""按四类采样方案生成六帧、三视角的 JSONL 索引，不解码或切分视频。

python -m qwen_training.build_sample_index --dataset-root /root/dataset/数据集 \
    --delta-dir /root/pt

标注放在 dataset-root/labels，输出放在 dataset-root/samples。
JSON 的 30FPS 起止帧均包含端点；计算区间时转换为左闭右开。
索引中的帧编号从 0 开始，空白前缀为 null，frame_mask 为 [3,6]。
仅 nowid 变化的跨阶段样本扩充：原有 k 条共享偏移窗口，再加 k 条区间内独立随机窗口。
"""

import argparse
from bisect import bisect_right
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random

from qwen_training.delta_to_progress import delta_to_progress


CAMERAS = {
    "top": "observation.images.camera_top",
    "left": "observation.images.camera_wrist_left",
    "right": "observation.images.camera_wrist_right",
}
KINDS = ("within_stage", "initial_padding", "cross_stage_changed", "cross_stage_same")


def state_of(subtask, task_ids):
    now = int(subtask["nowid"])
    rest = sorted(int(x) for x in subtask["restid"].split(",") if x.strip())
    return {"nowid": now, "restid": rest, "finishid": sorted(task_ids - {now} - set(rest))}


def candidate_windows(starts, width):
    """枚举首区间的整数帧；各区间共享偏移，取区间内最近的原始帧。

    使用 Fraction 计算帧单位的区间，避免浮点误差把右端点算进去。
    非整数时间戳只能近似到 30fps 帧网格，不改变区间或阶段归属。
    """
    if width <= 0:
        return []
    bounds = [(math.ceil(start), math.ceil(start + width) - 1) for start in starts]
    if any(low > high for low, high in bounds):
        return []
    windows = []
    for first in range(bounds[0][0], bounds[0][1] + 1):
        offset = first - starts[0]
        frames = tuple(
            min(high, max(low, math.floor(start + offset + Fraction(1, 2))))
            for start, (low, high) in zip(starts, bounds)
        )
        windows.append(frames)
    return list(dict.fromkeys(windows))


def independent_windows(starts, width, original, rng):
    """六个区间各自随机选帧，新增与原窗口等量的唯一窗口；组合不足则取尽。"""
    if not original:
        return []
    bounds = [(math.ceil(start), math.ceil(start + width)) for start in starts]
    available = math.prod(high - low for low, high in bounds) - len(original)
    count = min(len(original), available)
    seen, windows = set(original), []
    while len(windows) < count:
        frames = tuple(rng.randrange(low, high) for low, high in bounds)
        if frames not in seen:
            seen.add(frames)
            windows.append(frames)
    return windows


def sample_groups(subtasks, augmentation_rng=None):
    """返回 (类别, 起始阶段下标, 组号, 候选窗口, 数量上限)。

    传入 augmentation_rng 时，对跨阶段 nowid 变化的边界补充独立随机窗口，
    group=0 为原窗口，group=1 为补充窗口；不传时可复现原采样方案。
    """
    for index, subtask in enumerate(subtasks):
        start = subtask["30FPSstart_frame"]
        width = Fraction(subtask["30FPSend_frame"] + 1 - start, 6)
        windows = candidate_windows([start + j * width for j in range(6)], width)
        yield "within_stage", index, 0, windows, 15

    first = subtasks[0]
    start = first["30FPSstart_frame"]
    width = Fraction(first["30FPSend_frame"] + 1 - start, 6)
    for k in range(1, 6):
        windows = candidate_windows([start + j * width for j in range(k)], width)
        windows = [(None,) * (6 - k) + frames for frames in windows]
        yield "initial_padding", 0, k, windows, 10

    for index, (a, b) in enumerate(zip(subtasks, subtasks[1:])):
        boundary = b["30FPSstart_frame"]
        length_a = a["30FPSend_frame"] + 1 - a["30FPSstart_frame"]
        length_b = b["30FPSend_frame"] + 1 - boundary
        # 30fps 下 0.2s = 6 帧；跨界两帧额外间隔 12 帧，即 0.4s。
        width = min(Fraction(length_a - 6, 5), Fraction(length_b - 6), Fraction(30))
        starts = [boundary - 6 - (5 - j) * width for j in range(5)] + [Fraction(boundary + 6)]
        windows = candidate_windows(starts, width)
        if len(windows) > 30:
            raise ValueError("跨阶段变更样本超过 30 个，请先确认采样算法或设计")
        yield "cross_stage_changed", index, 0, windows, len(windows)
        if augmentation_rng is not None:
            extra = independent_windows(starts, width, windows, augmentation_rng)
            yield "cross_stage_changed", index, 1, extra, len(extra)

        width = min(Fraction(length_a - 6, 5), Fraction(length_b - 6, 5), Fraction(30))
        starts = [boundary - 6 - (4 - j) * width for j in range(4)]
        starts += [boundary + 6 + j * width for j in range(5)]
        for group in range(4):
            windows = candidate_windows(starts[group:group + 6], width)
            yield "cross_stage_same", index, group + 1, windows, 15


def choose_windows(windows, count, rng):
    """所有类别均不放回；候选不足时取全部可用窗口，不补足。"""
    return rng.sample(windows, min(count, len(windows)))


def build_index(dataset_root, delta_dir, seed=0):
    rng = random.Random(seed)
    # 扩充使用独立随机数生成器，保留所有原有窗口的选择结果。
    augmentation_rng = random.Random(seed + 1)
    label_dir = dataset_root / "labels"
    output_dir = dataset_root / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(p for p in label_dir.iterdir() if p.suffix.lower() == ".json")
    if not label_files:
        raise FileNotFoundError(f"没有标注：{label_dir}")

    counts = Counter(dict.fromkeys(KINDS, 0))
    unique = {kind: set() for kind in KINDS}
    targets = {kind: Counter() for kind in KINDS}
    pairs = {kind: Counter() for kind in KINDS}
    stages, groups, sources = Counter(), [], []
    episodes = 0
    progress_min, progress_max = math.inf, -math.inf
    pending = output_dir / "samples.jsonl.tmp"
    with pending.open("w", encoding="utf-8") as output:
        for label_path in label_files:
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
            task_ids = {int(task["id"]) for task in annotation["taskid"]}
            delta_path = delta_dir / f"{label_path.stem}-delta.txt"
            delta_bytes = delta_path.read_bytes()
            deltas = [float(line.split(maxsplit=1)[0]) for line in delta_bytes.decode().splitlines()]
            if any(not math.isfinite(value) or value < 0 for value in deltas):
                raise ValueError(f"delta 必须为有限非负数：{delta_path}")
            progress = delta_to_progress(deltas, annotation)
            if min(progress) < -1e-8 or max(progress) > 1 + 1e-8:
                raise ValueError(f"进度超出 [0,1]：{label_path}")
            progress_min = min(progress_min, min(progress))
            progress_max = max(progress_max, max(progress))
            videos = {
                view: str(dataset_root / "videos" / name / "chunk-000" / f"{label_path.stem}.mp4")
                for view, name in CAMERAS.items()
            }
            for path in videos.values():
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            sources.append({
                "video_id": label_path.stem, "label_frames": len(deltas),
                "annotation_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                "delta_sha256": hashlib.sha256(delta_bytes).hexdigest(),
            })

            for task_index, task in enumerate(annotation["tasks"]):
                for episode in task["episodes"]:
                    episodes += 1
                    subtasks = episode["subtasks"]
                    starts = [s["30FPSstart_frame"] for s in subtasks]
                    states = [state_of(s, task_ids) for s in subtasks]
                    stages.update(s["nowid"] for s in states)
                    for kind, segment, group, candidates, count in sample_groups(subtasks, augmentation_rng):
                        augmented = kind == "cross_stage_changed" and group == 1
                        chosen = candidates if augmented else choose_windows(candidates, count, rng)
                        groups.append({
                            "video_id": label_path.stem, "task_index": task_index,
                            "episode": episode["episode"], "sample_type": kind,
                            "segment_index": segment, "group": group,
                            "candidates": len(candidates), "samples": len(chosen),
                            "unique_samples": len(set(chosen)),
                            "sampling_method": "independent_intervals" if augmented else "shared_offset",
                        })
                        for frames in chosen:
                            valid = [frame for frame in frames if frame is not None]
                            if any(a >= b for a, b in zip(valid, valid[1:])):
                                raise ValueError("样本的有效帧必须按时间严格递增")
                            if valid[0] < starts[0] or valid[-1] > subtasks[-1]["30FPSend_frame"]:
                                raise ValueError("采样超出当前 episode")
                            # k=1 时没有倒数第二个真实帧，输入取首阶段状态。
                            before = 0 if frames[-2] is None else bisect_right(starts, frames[-2]) - 1
                            after = bisect_right(starts, frames[-1]) - 1
                            input_state, target_state = states[before], states[after]
                            changed = input_state["nowid"] != target_state["nowid"]
                            if changed != (kind == "cross_stage_changed"):
                                raise ValueError(f"样本类型与状态不符：{label_path.name} {frames}")
                            mask = [frame is not None for frame in frames]
                            record = {
                                "sample_id": f"{dataset_root.name}/{sum(counts.values()):07d}",
                                "dataset": dataset_root.name, "video_id": label_path.stem,
                                "task_index": task_index, "episode": episode["episode"],
                                "sample_type": kind, "group": group,
                                "sampling_method": "independent_intervals" if augmented else "shared_offset",
                                "video_paths": videos, "frame_indices": frames,
                                "frame_mask": [mask, mask, mask], "input_state": input_state,
                                "target": {**target_state, "progress": round(progress[frames[-1]], 8)},
                            }
                            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                            counts[kind] += 1
                            unique[kind].add((label_path.stem, task_index, episode["episode"], frames))
                            targets[kind][target_state["nowid"]] += 1
                            key = json.dumps([input_state, target_state], separators=(",", ":"))
                            pairs[kind][key] += 1

    result = {
        "dataset": dataset_root.name, "dataset_root": str(dataset_root),
        "settings": {
            "seed": seed, "fps": 30, "num_frames": 6, "camera_order": list(CAMERAS),
            "frame_index_base": 0, "padding_frame": None, "frame_mask_shape": [3, 6],
            "boundary_margin_seconds": 0.2, "cross_boundary_extra_seconds": 0.4,
            "time_rounding": "原窗口共享偏移后取区间内最近整数帧；补充窗口在各区间整数帧中独立随机选取",
            "short_within_stage": "不放回；普通最多 15 个 / 特殊每组最多 10 个，候选不足时取全部，不补足",
            "cross_stage": "变更样本保留原有 k 条（k<=30），再补 k 条各区间独立随机且不重复的窗口；不变样本每组最多 15",
            "augmentation": "仅 cross_stage_changed；随机种子 seed+1；唯一组合不足时取尽，不补重复窗口",
            "progress": "现有 delta_to_progress 算法，以最新标注逐 episode 计算",
            "split": "尚未划分训练/验证集；后续按原视频划分，避免窗口泄漏",
        },
        "summary": {
            "videos": len(label_files), "episodes": episodes, "segments": sum(stages.values()),
            "total_samples": sum(counts.values()), "samples_by_type": dict(counts),
            "unique_samples_by_type": {kind: len(unique[kind]) for kind in KINDS},
            "repeated_samples_by_type": {kind: counts[kind] - len(unique[kind]) for kind in KINDS},
            "zero_sample_groups": sum(g["samples"] == 0 for g in groups),
            "independent_cross_stage_samples": sum(
                g["samples"] for g in groups if g["sampling_method"] == "independent_intervals"
            ),
            "progress_range": [progress_min, progress_max],
        },
        "stage_occurrences": dict(sorted(stages.items())),
        "target_nowid_by_type": {kind: dict(sorted(targets[kind].items())) for kind in KINDS},
        "label_pairs_by_type": {
            kind: [{"input": json.loads(key)[0], "target": json.loads(key)[1], "count": n}
                   for key, n in sorted(pairs[kind].items())] for kind in KINDS
        },
        "sources": sources, "groups": groups,
    }
    pending.replace(output_dir / "samples.jsonl")
    (output_dir / "statistics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = build_index(args.dataset_root.resolve(), args.delta_dir.resolve(), args.seed)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
