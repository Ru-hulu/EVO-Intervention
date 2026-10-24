# ruff: noqa: RUF001, RUF002
"""把 RMBench 的分段标注转成逐帧的流式进度标签（proposal_progress 3.2.1、3.5、5.3）。

python -m qwen_training.datasets.rmbench --root /path/to/rmbench/data --out /path/to/rmbench_labels \
    --source-fps 16.67

RMBench 每 15 个仿真步（步长 1/250 s）存一帧，标称约 16.67 Hz，move 起止处另有额外帧。
每条 episode 输出一个 JSON：out/<task>/episode<N>.json。图像不复制，训练时按帧号从原 HDF5 的
observation/<camera>/rgb 读取。

读图注意：RMBench 的 JPEG 是 RoboTwin 旧格式，写入时把 RGB 数组直接交给了 cv2.imencode。
用 cv2.imdecode 读出来正好是 RGB；用 PIL、ffmpeg、浏览器等标准解码器读，红蓝通道是反的，
需要再交换一次。缓冲区里带 COM 注释 b"XPL-RGB1" 的是标准 RGB JPEG，不用交换
（见 RMBench 仓库 data/decode_image_bit.py）。cover_blocks、classify_blocks 等任务靠颜色判断，
读错会直接影响标签对应的画面。

字段：
- 模型输入：instruction（任务通用指令）、steps（tasks/<task>.py 里手写的通用步骤文字）。
  不包含本条 episode 的次数和只有记忆才知道的答案。
- demo：这条 episode 被抽作示范时提供的关键帧 [{step, frame}]，step 为 None 表示成功时的画面。
  固定流程给每个实例；次数由场景决定的给每项第一次；试错给每项第一次和成功画面。
- 监督：trace（实例的项、第几次、起止帧、原始标注文字）、step_counts（本条每项实际做了几次）、
  frames（按 sample_fps 取的原始帧，末帧一定包含在内）。
  frames 中 stage 为所在实例的项；counts[j] 为第 j 项已完成的次数（实例最后一帧起计入）；
  progress = (已完成实例数 + 当前实例内已经过的帧比例) / 这条 episode 的实例总数，三类任务相同；
  标注使用整条 episode 的信息，试错任务的实例总数即实际尝试次数。p 从 0 连续涨到 1。
RMBench 只有成功示范，counts 只增不减。
"""

import argparse
import json
from itertools import groupby
from pathlib import Path

from qwen_training.datasets.rmbench.spec import FIXED, SEARCH
from qwen_training.datasets.rmbench.tasks import TASKS

CAMERAS = ("head_camera", "left_camera", "right_camera")
MAX_SLOTS = 16
SAMPLE_FPS = 5


def instances(segments, spec):
    """segments: [[文字, 帧数], ...] → [(原始文字, 起始帧, 结束帧)]，帧号含两端。"""
    spans, start = [], 0
    for text, length in segments:
        spans.append((text, start, start + length - 1))
        start += length
    result = []
    for text, run in groupby(spans, key=lambda span: span[0]):
        run = list(run)
        if text in spec.merge_into_previous and result:
            result[-1] = (result[-1][0], result[-1][1], run[-1][2])
            continue
        size = spec.segments_per_instance or len(run)
        if len(run) % size:
            raise ValueError(f"“{text}”连续 {len(run)} 段，不能按每 {size} 段一个实例切分")
        result += [(text, run[i][1], run[i + size - 1][2]) for i in range(0, len(run), size)]
    return result


def step_of(text, order, spec):
    if spec.kind == FIXED:
        return order
    for prefix, step in spec.step_prefixes:
        if text.startswith(prefix):
            return step
    raise ValueError(f"标注“{text}”不匹配任何项")


def sample_frames(total, source_fps, sample_fps=SAMPLE_FPS):
    """原始帧号的采样网格；末帧总是包含在内，保证最后一个实例能被计入。"""
    stride = source_fps / sample_fps
    frames, k = [], 0
    while round(k * stride) < total:
        frames.append(round(k * stride))
        k += 1
    if frames[-1] != total - 1:
        frames.append(total - 1)
    return frames


def demo_keyframes(trace, spec):
    if spec.kind == FIXED:
        return [{"step": item["step"], "frame": item["end"]} for item in trace]
    firsts = [{"step": item["step"], "frame": item["end"]} for item in trace if item["instance"] == 1]
    firsts.sort(key=lambda key: key["step"])
    if spec.kind == SEARCH:
        firsts.append({"step": None, "frame": trace[-1]["end"]})
    return firsts


def episode_labels(segments, spec, source_fps, sample_fps=SAMPLE_FPS):
    spans = instances(segments, spec)
    if spec.kind == FIXED and len(spans) != len(spec.steps):
        raise ValueError(f"固定流程应有 {len(spec.steps)} 个实例，实际 {len(spans)} 个")
    trace, step_counts = [], [0] * len(spec.steps)
    for order, (text, start, end) in enumerate(spans):
        step = step_of(text, order, spec)
        step_counts[step] += 1
        trace.append({"step": step, "instance": step_counts[step], "start": start, "end": end, "annotation": text})
    capacity = sum(spec.max_counts) if spec.kind == SEARCH else len(trace)
    if capacity > MAX_SLOTS:
        raise ValueError(f"需要 {capacity} 个槽位，超过 {MAX_SLOTS} 个")
    if spec.kind == SEARCH and any(n > m for n, m in zip(step_counts, spec.max_counts, strict=True)):
        raise ValueError(f"尝试次数 {step_counts} 超过上限 {list(spec.max_counts)}")
    frames = []
    for index in sample_frames(trace[-1]["end"] + 1, source_fps, sample_fps):
        current = next(item for item in trace if item["start"] <= index <= item["end"])
        counts = [0] * len(spec.steps)
        for item in trace:
            counts[item["step"]] += index >= item["end"]
        length = current["end"] - current["start"] + 1
        elapsed = 0 if index >= current["end"] else (index - current["start"]) / length
        progress = round((sum(counts) + elapsed) / len(trace), 6)
        frames.append({"index": index, "stage": current["step"], "counts": counts, "progress": progress})
    return {"steps": list(spec.steps), "step_counts": step_counts,
            "max_counts": list(spec.max_counts) or None, "trace": trace,
            "demo": demo_keyframes(trace, spec), "frames": frames}


def frame_count(path):
    import h5py  # 只在读取真实数据时需要。

    with h5py.File(path, "r") as f:
        counts = {len(f[f"observation/{camera}/rgb"]) for camera in CAMERAS}
    if len(counts) != 1:
        raise ValueError(f"{path}: 各相机帧数不一致 {counts}")
    return counts.pop()


def build(root, out, source_fps, tasks=None, count_frames=frame_count):
    root, out = Path(root), Path(out)
    written = 0
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        task = task_dir.name
        if tasks and task not in tasks:
            continue
        spec = TASKS[task]
        demo = task_dir / "demo_clean"
        annotation = json.loads((demo / "language_annotation.json").read_text())
        for key, segments in annotation.items():
            episode = int(key.removeprefix("episode_"))
            source = demo / "data" / f"episode{episode}.hdf5"
            total = sum(length for _, length in segments)
            if count_frames(source) != total:
                raise ValueError(f"{source}: 分段总帧数 {total} 与 HDF5 帧数不一致")
            instruction = json.loads((demo / "instructions" / f"episode{episode}.json").read_text())["seen"][0]
            try:
                labels = episode_labels(segments, spec, source_fps)
            except ValueError as error:
                raise ValueError(f"{task} episode {episode}: {error}") from error
            record = {"task": task, "episode": episode, "kind": spec.kind, "source": str(source.relative_to(root)),
                      "source_fps": source_fps, "cameras": list(CAMERAS), "instruction": instruction, **labels}
            target = out / task / f"episode{episode}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record, ensure_ascii=False))
            written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="RMBench 的 data 目录（其下每个任务一个子目录）")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-fps", type=float, required=True, help="RMBench 原始帧率，标称 250/15≈16.67")
    parser.add_argument("--tasks", nargs="*", help="只处理这些任务，默认全部")
    args = parser.parse_args()
    print(f"wrote {build(args.root, args.out, args.source_fps, args.tasks)} episodes")
