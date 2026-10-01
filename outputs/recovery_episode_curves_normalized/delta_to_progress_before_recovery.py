#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""根据每帧的 delta 和 description，计算整条视频的归一化进度。

    python delta_to_progress.py

默认读取 /root/pt/file-xxx-delta.txt，每行必须是「delta 阶段描述」，
生成 /root/pt/file-xxx-progress.txt，每行是「进度 阶段描述」。
连续相同描述视为一个子任务；不连续的同名子任务分别计数，无需 JSON。
"""

import argparse
from itertools import accumulate
from itertools import pairwise
import math
from pathlib import Path


def delta_to_progress(deltas: list[float], descriptions: list[str]) -> list[float]:
    """输入等长的逐帧 delta 和描述，返回逐帧进度。

    输入 delta 应为有限的非负数；首帧没有前驱，固定为 0。
    s、e 由描述变化的位置确定，包含首尾帧；权重按公式使用 e-s。
    r_i = delta_i / sum(delta[s:e+1])，w = clip(K*(e-s)/T, 0.75, 1.25)。
    p_i = sum(w*r 到第 i 帧) / sum(w*r 到最后一帧)。
    全零子任务不贡献进度；整条轨迹无有效变化时，无法归一化，抛出异常。
    """
    if not deltas or len(deltas) != len(descriptions):
        raise ValueError("delta 和描述必须非空且逐帧对应")
    if any(not description.strip() or description.strip() == "未标注" for description in descriptions):
        raise ValueError("每一帧都必须有有效的阶段描述")
    deltas = list(deltas)
    deltas[0] = 0.0
    total_frames = len(deltas)

    # 每次描述变化即开始一个新子任务；区间右侧 stop 不包含在内。
    boundaries = [0]
    boundaries.extend(i for i in range(1, total_frames) if descriptions[i] != descriptions[i - 1])
    boundaries.append(total_frames)
    task_count = len(boundaries) - 1

    growth = [0.0] * total_frames
    for start, stop in pairwise(boundaries):
        end = stop - 1
        segment = deltas[start:stop]
        delta_sum = math.fsum(segment)
        if delta_sum == 0:
            continue
        weight = min(1.25, max(0.75, task_count * (end - start) / total_frames))
        growth[start:stop] = [weight * (delta / delta_sum) for delta in segment]

    cumulative = list(accumulate(growth))
    total = cumulative[-1]
    if not math.isfinite(total) or total <= 0:
        raise ValueError("整条轨迹没有可归一化的有效 delta")
    return [value / total for value in cumulative]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-dir", type=Path, default=Path("/root/pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/pt"))
    args = parser.parse_args()

    delta_files = sorted(args.delta_dir.glob("file-*-delta.txt"))
    if not delta_files:
        raise FileNotFoundError(f"{args.delta_dir} 下没有 file-*-delta.txt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for delta_path in delta_files:
        stem = delta_path.name.removesuffix("-delta.txt")
        deltas, descriptions = [], []
        for line in delta_path.read_text(encoding="utf-8").splitlines():
            value, description = line.split(maxsplit=1)
            deltas.append(float(value))
            descriptions.append(description.strip())
        progress = delta_to_progress(deltas, descriptions)

        output_path = args.output_dir / f"{stem}-progress.txt"
        with output_path.open("w", encoding="utf-8") as output:
            for value, description in zip(progress, descriptions, strict=True):
                output.write(f"{value:.6f} {description}\n")
        print(f"完成 {output_path.name}：{len(progress)} 帧 / {len(progress)} 行")


if __name__ == "__main__":
    main()
