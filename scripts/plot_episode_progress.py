#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""从有 JSON 和 delta 的 episode 中不放回随机抽样，并绘制逐帧进度。"""

import argparse
import json
import math
from pathlib import Path
import random
import sys

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.delta_to_progress import episode_to_progress
from scripts.delta_to_progress import frame_descriptions
from scripts.find_rollback_episodes import find_rollback_episodes

mpl.use("Agg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta-dir", type=Path, default=Path("/root/pt"))
    parser.add_argument("--label-dir", type=Path, default=Path("/root/lable"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--min-subtasks", type=int, default=0, help="仅从子任务数不少于此值的 episode 中抽样")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--rollback-only", action="store_true", help="扫描所有 JSON，绘制全部回滚 episode")
    args = parser.parse_args()
    if args.min_subtasks < 0 or args.count < 1 or args.fps <= 0:
        parser.error("--min-subtasks 必须非负，--count 和 --fps 必须为正数")
    if args.rollback_only and args.min_subtasks:
        parser.error("--min-subtasks 仅用于随机抽样，不能与 --rollback-only 同时使用")
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollback_keys = None
    scan = None
    if args.rollback_only:
        scan = find_rollback_episodes(args.label_dir)
        (args.output_dir / "rollback_scan.json").write_text(
            json.dumps(scan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rollback_keys = {(item["file"], item["episode"]) for item in scan["rollback_episodes"]}

    videos, pool, task_names = {}, [], {}
    for delta_path in sorted(args.delta_dir.glob("file-*-delta.txt")):
        stem = delta_path.name.removesuffix("-delta.txt")
        label_path = args.label_dir / f"{stem}.JSON"
        if not label_path.exists():
            label_path = args.label_dir / f"{stem}.json"
        if not label_path.exists():
            continue
        annotation = json.loads(label_path.read_text(encoding="utf-8"))
        delta = [float(row.split(maxsplit=1)[0]) for row in delta_path.read_text(encoding="utf-8").splitlines()]
        descriptions = frame_descriptions(annotation, len(delta))
        videos[stem] = annotation, delta, descriptions
        task_names.update({task["id"]: task["name"] for task in annotation["taskid"]})
        for task in annotation["tasks"]:
            pool.extend(
                (stem, episode)
                for episode in task["episodes"]
                if len(episode["subtasks"]) >= args.min_subtasks
                and (rollback_keys is None or (stem, episode["episode"]) in rollback_keys)
            )

    # 只抽样一次，保存 seed 和名单，便于重现。
    if args.rollback_only:
        missing = rollback_keys - {(stem, episode["episode"]) for stem, episode in pool}
        if missing:
            raise FileNotFoundError(f"已保存完整扫描结果；以下回滚 episode 缺少 delta：{sorted(missing)}")
        selected = pool
        if not selected:
            print("未发现回滚 episode，扫描结果已保存。")
            return
    else:
        if len(pool) < args.count:
            parser.error(f"符合条件且已有 delta 的 episode 仅 {len(pool)} 个，无法抽取 {args.count} 个")
        selected = random.Random(seed).sample(pool, args.count)
    selected.sort(key=lambda item: (item[0], item[1]["episode"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for stem, episode in selected:
        annotation, delta, descriptions = videos[stem]
        subtasks = episode["subtasks"]
        task_ids = [task["id"] for task in annotation["taskid"]]
        progress, cumulative = episode_to_progress(delta, subtasks, task_ids)
        start, end = subtasks[0]["30FPSstart_frame"], subtasks[-1]["30FPSend_frame"]
        values = np.array(progress)
        drops = np.flatnonzero(np.diff(values) < -1e-10) + 1
        summary = {
            "file": stem,
            "episode": episode["episode"],
            "start_frame": start,
            "end_frame": end,
            "frames": len(progress),
            "subtask_count": len(subtasks),
            "task_sequence": [subtask["nowid"] for subtask in subtasks],
            "video_start_seconds": start / args.fps,
            "video_end_seconds_exclusive": (end + 1) / args.fps,
            "rollback_count": len(drops),
            "minimum": min(progress),
            "maximum": max(progress),
            "final": progress[-1],
            "rollback_frames": [start + int(i) for i in drops],
            "rollback_times_seconds": [int(i) / args.fps for i in drops],
            "rollback_drop_percentage_points": [float(values[i - 1] - values[i]) * 100 for i in drops],
        }
        output_path = args.output_dir / f"{stem}-episode-{episode['episode']}-progress.txt"
        with output_path.open("w", encoding="utf-8") as output:
            output.write("frame\tepisode_seconds\tscaled_cumulative\tprogress\tdescription\n")
            for i, frame in enumerate(range(start, end + 1)):
                output.write(
                    f"{frame}\t{i / args.fps:.6f}\t{cumulative[i]:.9f}\t{progress[i]:.9f}\t{descriptions[frame]}\n"
                )
        results.append((summary, subtasks, values, drops))

    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#64748b",
            "ytick.color": "#64748b",
        }
    )
    rows = math.ceil(len(results) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3 * rows + 2.2), squeeze=False, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.965, bottom=0.09, top=0.79, hspace=0.5, wspace=0.15)
    fig.suptitle(
        f"{len(task_names)} 阶段任务进度 · "
        + (f"全部 {len(results)} 个回滚 episode" if args.rollback_only else f"随机抽取 {len(results)} 个 episode"),
        x=0.085,
        y=0.975,
        ha="left",
        fontsize=21,
        color="#0f172a",
    )
    fig.text(
        0.085,
        0.910,
        (
            f"扫描 {scan['json_files']} 份 JSON / {scan['episodes']} 个 episode · 红线为回滚"
            if args.rollback_only
            else f"已回传 {len(videos)} 个视频 · 候选 {len(pool)} 个 episode"
            + (f"（Subtask ≥ {args.min_subtasks}）" if args.min_subtasks else "")
            + f" · 随机种子 {seed}"
        )
        + f" · 原视频 {args.fps:g} fps",
        color="#64748b",
        fontsize=11,
    )
    colors = {task_id: plt.get_cmap("tab10")(i % 10) for i, task_id in enumerate(task_names)}
    fig.legend(
        handles=[
            Patch(facecolor=colors[task_id], alpha=0.22, label=f"{task_id} · {name}")
            for task_id, name in task_names.items()
        ],
        loc="upper left",
        bbox_to_anchor=(0.078, 0.907),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    xmax = max(5, math.ceil(max((len(item[2]) - 1) / args.fps for item in results) / 5) * 5)
    for axis, (summary, subtasks, values, drops) in zip(axes.flat, results, strict=False):
        start = summary["start_frame"]
        time = np.arange(len(values)) / args.fps
        for subtask in subtasks:
            left = (subtask["30FPSstart_frame"] - start) / args.fps
            right = (subtask["30FPSend_frame"] + 1 - start) / args.fps
            axis.axvspan(left, right, color=colors[subtask["nowid"]], alpha=0.09, zorder=0)
            axis.text((left + right) / 2, 1.06, subtask["nowid"], ha="center", va="center", color="#475569", fontsize=9)
        axis.axhline(1, color="#94a3b8", linestyle="--", linewidth=0.8)
        axis.plot(time, values, color="#2463eb", linewidth=1.6, zorder=3)
        axis.scatter(time[-1], values[-1], s=17, color="#2463eb", zorder=4)
        for i in drops:
            axis.axvline(time[i], color="#dc4b38", linestyle="--", linewidth=1.2)
            axis.scatter([time[i - 1], time[i]], [values[i - 1], values[i]], color="#dc4b38", s=22, zorder=5)
        title = (
            f"{summary['file']} / episode {summary['episode']}"
            f"  ·  {len(subtasks)} 个 Subtask / {len(drops)} 次回滚"
        )
        axis.set_title(title, loc="left", fontsize=11, color="#0f172a", pad=10)
        axis.set_xlim(0, xmax)
        axis.set_ylim(0, 1.13)
        axis.set_xticks(np.linspace(0, xmax, 6))
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        axis.tick_params(length=0, pad=6, labelbottom=True)
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    for axis in list(axes.flat)[len(results) :]:
        axis.set_visible(False)
    fig.supxlabel("episode 内时间（秒） · 红色虚线为进度回滚点", y=0.015, fontsize=12, color="#334155")
    fig.supylabel("任务进度 p(t)", x=0.018, fontsize=12, color="#334155")
    plot_name = "rollback_progress" if args.rollback_only else "sampled_progress"
    fig.savefig(args.output_dir / f"{plot_name}.png", dpi=190, facecolor="white")
    fig.savefig(args.output_dir / f"{plot_name}.pdf", facecolor="white")
    plt.close(fig)
    report = {
        "mode": "rollback" if args.rollback_only else "random",
        "seed": None if args.rollback_only else seed,
        "candidate_episodes": len(pool),
        "available_videos": len(videos),
        "available_video_files": list(videos),
        "min_subtasks": args.min_subtasks,
        "candidate_pool": [
            {"file": stem, "episode": episode["episode"], "subtask_count": len(episode["subtasks"])}
            for stem, episode in pool
        ],
        "label_dir": str(args.label_dir.resolve()),
        "delta_dir": str(args.delta_dir.resolve()),
        "fps": args.fps,
        "task_names": task_names,
        "selected": [item[0] for item in results],
    }
    (args.output_dir / "selection.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
