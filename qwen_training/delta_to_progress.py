#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""根据 delta 和 JSON 标注，逐 episode 计算支持失败回滚的进度。

    python delta_to_progress.py

默认读取 /root/pt/file-xxx-delta.txt 和 /root/lable/file-xxx.JSON，
生成 /root/pt/file-xxx-progress.txt，每行是「进度 阶段描述」。
阶段描述从当前 JSON 生成，空隙帧按就近原则补齐，不沿用 delta 文件的旧描述。
直接使用 JSON 的 30FPSstart_frame / 30FPSend_frame，包含首尾帧。
同一任务 ID 以最后一次执行的 delta 总和为贡献目标；每个 episode 独立归一化。
最后一次执行视为成功，成功与否由标注保证。视频开头尚未进入 episode 时进度为 0，
episode 结束后的空白范围保持 1，进入下一个 episode 时重新计算。
"""

import argparse
from itertools import pairwise
import json
import math
from pathlib import Path


def episode_to_progress(deltas: list[float], subtasks: list[dict], task_ids: list[str]):
    """返回一个 episode 的 (归一化进度, 缩放并回滚后的累计值)。

    deltas 使用整条视频的帧索引，返回的两个列表只包含当前 episode 的帧。
    输入 delta 应为有限非负数，子任务帧区间必须连续且不重叠。
    每次出现的 subtask 分别求和；同一 ID 的归一化目标只保留最后一次。
    全零段无法按本算法缩放，直接报错，避免生成无依据的进度。
    """
    frame_start = subtasks[0]["30FPSstart_frame"]
    frame_end = subtasks[-1]["30FPSend_frame"]
    subtask_con = []
    norm_fact = dict.fromkeys(task_ids, 0.0)

    next_frame = frame_start
    for subtask in subtasks:
        start, end = subtask["30FPSstart_frame"], subtask["30FPSend_frame"]
        if start != next_frame or not 0 <= start <= end < len(deltas):
            raise ValueError(f"子任务帧范围不连续或越界：{start}–{end}")
        total = math.fsum(deltas[start : end + 1])
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"子任务 {subtask['nowid']}（帧 {start}–{end}）的 delta 总和必须大于 0")
        subtask_con.append(total)
        norm_fact[subtask["nowid"]] = total
        next_frame = end + 1

    p = [0.0] * (frame_end - frame_start + 1)
    contri = dict.fromkeys(task_ids, 0.0)
    pre_r = set()
    finishid = set()

    # 第二遍直接使用每段起止帧，避免沿用第一遍结束后的 frame。
    for subtask, total in zip(subtasks, subtask_con, strict=True):
        start, end = subtask["30FPSstart_frame"], subtask["30FPSend_frame"]
        nowid = subtask["nowid"]
        rest = {value.strip() for value in subtask["restid"].split(",") if value.strip()}
        now_r = rest | {nowid}
        fail_set = now_r - pre_r

        # 当前段首帧先记录扣减，下面用 += 继承上一帧，保留这次回滚。
        p[start - frame_start] -= math.fsum(contri[i] for i in fail_set)
        finishid.difference_update(fail_set)
        for task_id in fail_set:
            contri[task_id] = 0.0

        for frame in range(start, end + 1):
            stepcon = deltas[frame] * norm_fact[nowid] / total
            offset = frame - frame_start
            previous = p[offset - 1] if offset else 0.0
            p[offset] += previous + stepcon
            contri[nowid] += stepcon

        finishid.add(nowid)
        pre_r = rest

    total = p[-1]
    if not math.isfinite(total) or total <= 0:
        raise ValueError("episode 的最终贡献必须大于 0，无法归一化")
    return [value / total for value in p], p


def delta_to_progress(deltas: list[float], annotation: dict) -> list[float]:
    """计算整条视频的逐帧进度，每个 episode 独立从头累计。"""
    task_ids = [task["id"] for task in annotation["taskid"]]
    episodes = [episode for task in annotation["tasks"] for episode in task["episodes"]]
    if not episodes:
        raise ValueError("JSON 中没有 episode")
    progress = [0.0] * len(deltas)
    previous_stop = 0
    for episode in episodes:
        subtasks = episode["subtasks"]
        start = subtasks[0]["30FPSstart_frame"]
        stop = subtasks[-1]["30FPSend_frame"] + 1
        if start < previous_stop:
            raise ValueError("episode 帧范围重叠或顺序错误")
        values, _ = episode_to_progress(deltas, subtasks, task_ids)
        if previous_stop:
            progress[previous_stop:start] = [1.0] * (start - previous_stop)
        progress[start:stop] = values
        previous_stop = stop
    progress[previous_stop:] = [1.0] * (len(deltas) - previous_stop)
    return progress


def frame_descriptions(annotation: dict, frame_count: int) -> list[str]:
    """按当前 JSON 生成逐帧描述；空隙就近补齐，等距离时取前一段。"""
    subtasks = [
        subtask for task in annotation["tasks"] for episode in task["episodes"] for subtask in episode["subtasks"]
    ]
    boundaries = [0]
    boundaries.extend(
        (left["30FPSend_frame"] + right["30FPSstart_frame"]) // 2 + 1 for left, right in pairwise(subtasks)
    )
    boundaries.append(frame_count)
    descriptions = []
    for subtask, start, stop in zip(subtasks, boundaries[:-1], boundaries[1:], strict=True):
        descriptions.extend([subtask["name"]] * (stop - start))
    return descriptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-dir", type=Path, default=Path("/root/pt"))
    parser.add_argument("--label-dir", type=Path, default=Path("/root/lable"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/pt"))
    args = parser.parse_args()

    delta_files = sorted(args.delta_dir.glob("file-*-delta.txt"))
    if not delta_files:
        raise FileNotFoundError(f"{args.delta_dir} 下没有 file-*-delta.txt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for delta_path in delta_files:
        stem = delta_path.name.removesuffix("-delta.txt")
        deltas = [float(line.split(maxsplit=1)[0]) for line in delta_path.read_text(encoding="utf-8").splitlines()]
        label_path = args.label_dir / f"{stem}.JSON"
        if not label_path.exists():
            label_path = args.label_dir / f"{stem}.json"
        annotation = json.loads(label_path.read_text(encoding="utf-8"))
        progress = delta_to_progress(deltas, annotation)
        descriptions = frame_descriptions(annotation, len(deltas))

        output_path = args.output_dir / f"{stem}-progress.txt"
        with output_path.open("w", encoding="utf-8") as output:
            for value, description in zip(progress, descriptions, strict=True):
                output.write(f"{value:.6f} {description}\n")
        print(f"完成 {output_path.name}：{len(progress)} 帧 / {len(progress)} 行")


if __name__ == "__main__":
    main()
