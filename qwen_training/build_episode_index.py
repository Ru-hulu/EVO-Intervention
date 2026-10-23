# ruff: noqa: RUF001, RUF002
"""按 5 Hz 生成逐 episode 的流式标签索引：每步的 [STG] 阶段、[CNT] 槽位完成状态和进度。

python -m qwen_training.build_episode_index --dataset-root /root/dataset/数据集 \
    --delta-dir /root/pt

标注放在 dataset-root/labels，输出放在 dataset-root/episodes（episodes.jsonl + statistics.json）。
每行一条 episode；帧号为 30 FPS 原视频的帧号（从 0 开始），取全局 6 帧网格，
与 downsample_videos_preserve_timestamps 的 5 FPS 视频第 k 帧 = 原第 6k 帧对齐。

标签语义（proposal_progress 3.5、5.3）：
- stage：该步所在子任务段的 nowid 在 steps 中的下标；回退后重做的段标为被重做的项。
  当前标注的段首尾相接，不产生“不在列表里”；错配 prompt 的负样本留给训练时构造。
- slot_state：该步每个槽位当前是否处于完成状态。段内状态与 build_sample_index.state_of
  一致：已完成 = 全部任务 − nowid − restid；某项在其段结束后的下一段才变为 1，
  被破坏时在出现破坏的段首变回 0。时间分辨率是标注段边界，无法定位段内的破坏时刻。
  最后一项在 episode 内始终为 0，episode 结束的成功由 s(t) 标签表达。
- progress：现有 delta_to_progress 结果，取该步对应的原视频帧。
当前数据每个任务 ID 只要求完成一次，槽位与 steps 一一对应（repeat 均为 0）。
"""

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from qwen_training.build_sample_index import CAMERAS
from qwen_training.delta_to_progress import delta_to_progress
from qwen_training.prompts import STAGE_PREREQUISITES, TASK_INTENT, TASK_PROMPT


SOURCE_FPS = 30
STEP_HZ = 5
FRAME_STRIDE = SOURCE_FPS // STEP_HZ


def step_frames(start, end, stride=FRAME_STRIDE):
    """[start, end] 内落在全局 stride 网格上的帧；不足一个网格点时退回 start。"""
    first = math.ceil(start / stride) * stride
    return list(range(first, end + 1, stride)) or [start]


def episode_labels(subtasks, task_ids, frames):
    """返回 (stage_ids, finished_sets)，均与 frames 等长；task_ids 为 int 集合。"""
    starts = [s["30FPSstart_frame"] for s in subtasks]
    states = []
    for subtask in subtasks:
        now = int(subtask["nowid"])
        rest = {int(x) for x in subtask["restid"].split(",") if x.strip()}
        if now not in task_ids or not rest <= task_ids:
            raise ValueError(f"nowid/restid 不在 taskid 中：{subtask}")
        states.append((now, frozenset(task_ids - {now} - rest)))
    stage_ids, finished = [], []
    for frame in frames:
        index = bisect_right(starts, frame) - 1
        if index < 0 or frame > subtasks[-1]["30FPSend_frame"]:
            raise ValueError(f"帧 {frame} 不在 episode 内")
        now, done = states[index]
        stage_ids.append(now)
        finished.append(done)
    return stage_ids, finished


def rollback_steps(slot_state):
    """每个出现 1→0 的步：(步下标, 变回 0 的槽位下标列表)。"""
    events = []
    for t in range(1, len(slot_state)):
        reset = [i for i, (a, b) in enumerate(zip(slot_state[t - 1], slot_state[t], strict=True)) if a and not b]
        if reset:
            events.append([t, reset])
    return events


def build_index(dataset_root, delta_dir, dependencies=STAGE_PREREQUISITES):
    label_dir = dataset_root / "labels"
    output_dir = dataset_root / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(p for p in label_dir.iterdir() if p.suffix.lower() == ".json")
    if not label_files:
        raise FileNotFoundError(f"没有标注：{label_dir}")

    episodes = steps_total = 0
    stage_counts, rollback_counts, lengths, sources = Counter(), Counter(), [], []
    pending = output_dir / "episodes.jsonl.tmp"
    with pending.open("w", encoding="utf-8") as output:
        for label_path in label_files:
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
            items = [{"id": int(task["id"]), "name": task["name"]} for task in annotation["taskid"]]
            task_ids = {item["id"] for item in items}
            if set(dependencies) != task_ids:
                raise ValueError(f"依赖关系的任务 ID 与标注不一致：{label_path.name}")
            index_of = {item["id"]: j for j, item in enumerate(items)}
            steps = [{**item, "depends_on": sorted(index_of[i] for i in dependencies[item["id"]]), "count": 1}
                     for item in items]
            delta_path = delta_dir / f"{label_path.stem}-delta.txt"
            delta_bytes = delta_path.read_bytes()
            deltas = [float(line.split(maxsplit=1)[0]) for line in delta_bytes.decode().splitlines()]
            if any(not math.isfinite(value) or value < 0 for value in deltas):
                raise ValueError(f"delta 必须为有限非负数：{delta_path}")
            progress = delta_to_progress(deltas, annotation)
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
                    subtasks = episode["subtasks"]
                    start, end = subtasks[0]["30FPSstart_frame"], subtasks[-1]["30FPSend_frame"]
                    frames = step_frames(start, end)
                    stage_ids, finished = episode_labels(subtasks, task_ids, frames)
                    slot_state = [[int(item["id"] in done) for item in items] for done in finished]
                    events = rollback_steps(slot_state)
                    record = {
                        "episode_id": f"{dataset_root.name}/{label_path.stem}/{task_index}/{episode['episode']}",
                        "dataset": dataset_root.name, "video_id": label_path.stem,
                        "task_index": task_index, "episode": episode["episode"],
                        "intent": TASK_INTENT, "task_prompt": TASK_PROMPT,
                        "video_paths": videos, "source_fps": SOURCE_FPS, "step_hz": STEP_HZ,
                        "first_frame": start, "last_frame": end,
                        "steps": steps,
                        "slots": [[j, 0] for j in range(len(steps))],  # (第 j 项, 第 r 次)。
                        "frame_indices": frames,
                        "stage": [index_of[i] for i in stage_ids],
                        "slot_state": slot_state,
                        "progress": [round(progress[frame], 8) for frame in frames],
                        "rollback_steps": events,
                    }
                    output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    episodes += 1
                    steps_total += len(frames)
                    lengths.append(len(frames))
                    stage_counts.update(stage_ids)
                    rollback_counts[len(events)] += 1

    result = {
        "dataset": dataset_root.name, "dataset_root": str(dataset_root),
        "settings": {
            "source_fps": SOURCE_FPS, "step_hz": STEP_HZ, "frame_grid": f"原视频帧号为 {FRAME_STRIDE} 的倍数",
            "camera_order": list(CAMERAS), "frame_index_base": 0,
            "dependencies": {str(k): sorted(v) for k, v in dependencies.items()},
            "slot_state": "已完成 = 全部任务 − nowid − restid；分辨率为标注段边界",
            "stage": "nowid 在 steps 中的下标",
            "progress": "现有 delta_to_progress 算法",
            "split": "尚未划分训练/验证集；后续按原视频划分",
        },
        "summary": {
            "videos": len(label_files), "episodes": episodes, "steps": steps_total,
            "steps_per_episode": [min(lengths, default=0), max(lengths, default=0)],
            "stage_steps_by_nowid": dict(sorted(stage_counts.items())),
            "episodes_by_rollback_count": dict(sorted(rollback_counts.items())),
        },
        "sources": sources,
    }
    pending.replace(output_dir / "episodes.jsonl")
    (output_dir / "statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_index(args.dataset_root.resolve(), args.delta_dir.resolve())
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
