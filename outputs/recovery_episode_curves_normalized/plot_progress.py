"""调用正式计算函数，绘制两个异常 episode；同时保存归一化前后的数值。"""
# ruff: noqa: RUF001, RUF002, RUF003

import argparse
import json
import math
from pathlib import Path
import sys
import textwrap

import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from scripts.delta_to_progress import episode_to_progress  # noqa: E402
from scripts.delta_to_progress import frame_descriptions  # noqa: E402

mpl.use("Agg")
FPS = 30
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label-dir", type=Path, required=True, help="当前 JSON 所在目录")
args = parser.parse_args()


def calculate(stem, episode_id):
    label_path = args.label_dir / f"{stem}.JSON"
    if not label_path.exists():
        label_path = args.label_dir / f"{stem}.json"
    annotation = json.loads(label_path.read_text())
    episode = next(e for t in annotation["tasks"] for e in t["episodes"] if e["episode"] == episode_id)
    subtasks = episode["subtasks"]
    rows = (ROOT / "data" / f"{stem}-delta.txt").read_text().splitlines()
    delta = [float(row.split(maxsplit=1)[0]) for row in rows]
    descriptions = frame_descriptions(annotation, len(delta))
    task_ids = [task["id"] for task in annotation["taskid"]]
    progress, cumulative = episode_to_progress(delta, subtasks, task_ids)
    start, end = subtasks[0]["30FPSstart_frame"], subtasks[-1]["30FPSend_frame"]
    norm_fact = {}
    stages = []
    for subtask in subtasks:
        first, last = subtask["30FPSstart_frame"], subtask["30FPSend_frame"]
        total = math.fsum(delta[first : last + 1])
        norm_fact[subtask["nowid"]] = total
        stages.append({"nowid": subtask["nowid"], "start_frame": first, "end_frame": last, "delta_sum": total})

    # 图中的下降点取自正式函数的输出，不另行实现回滚计算。
    events = []
    for index in np.flatnonzero(np.diff(progress) < -1e-10) + 1:
        frame = start + int(index)
        stage_index = next(i for i, stage in enumerate(stages) if stage["start_frame"] == frame)
        subtask = subtasks[stage_index]
        now_r = set(filter(None, subtask["restid"].split(","))) | {subtask["nowid"]}
        pre_r = set(filter(None, subtasks[stage_index - 1]["restid"].split(",")))
        stepcon = delta[frame] * norm_fact[subtask["nowid"]] / stages[stage_index]["delta_sum"]
        events.append(
            {
                "frame": frame,
                "reset_ids": sorted(now_r - pre_r),
                "episode_seconds": (frame - start) / FPS,
                "source_seconds": frame / FPS,
                "before": progress[index - 1],
                "after": progress[index],
                "before_scaled": cumulative[index - 1],
                "after_scaled": cumulative[index],
                "removed_scaled": cumulative[index - 1] + stepcon - cumulative[index],
            }
        )

    summary = {
        "file": stem,
        "episode": episode_id,
        "start_frame": start,
        "end_frame": end,
        "frame_count": len(progress),
        "final_scaled": cumulative[-1],
        "maximum": max(progress),
        "target_contributions": norm_fact,
        "task_weights": {key: value / cumulative[-1] for key, value in norm_fact.items()},
        "events": events,
        "stages": stages,
    }
    with (ROOT / f"{stem}-episode-{episode_id}-progress.txt").open("w", encoding="utf-8") as output:
        output.write("frame\tepisode_seconds\tdelta\tscaled_cumulative\tprogress\tdescription\n")
        for index, frame in enumerate(range(start, end + 1)):
            description = descriptions[frame]
            output.write(
                f"{frame}\t{index / FPS:.6f}\t{delta[frame]:.6f}\t"
                f"{cumulative[index]:.9f}\t{progress[index]:.9f}\t{description}\n"
            )
    return summary, subtasks, np.array(progress), np.array(cumulative)


font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
plt.rcParams.update(
    {
        "font.family": "Arial Unicode MS",
        "font.size": 11,
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
results = [calculate("file-001", 5), calculate("file-005", 4)]
for mode in ("normalized", "scaled"):
    normalized = mode == "normalized"
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.085, top=0.855, hspace=0.32)
    title = "新算法：两个异常 episode 的进度曲线" if normalized else "新算法：缩放并回滚后的累计贡献"
    fig.suptitle(title, x=0.09, y=0.975, ha="left", fontsize=22, color="#0f172a")
    subtitle = "段内 delta 按该 ID 最后一次执行的总贡献缩放 → FailSet 回滚"
    subtitle += " → episode 终值归一化" if normalized else "（尚未除以 episode 终值）"
    fig.text(0.09, 0.925, subtitle, color="#475569", fontsize=11.5)
    fig.text(
        0.09,
        0.89,
        "灰色分区表示子任务；红线表示回滚；回滚首帧已计入该帧新增贡献；原视频 30 fps",
        color="#64748b",
        fontsize=10.5,
    )

    for axis, (summary, subtasks, progress, cumulative) in zip(axes, results, strict=True):
        values = progress if normalized else cumulative
        start = summary["start_frame"]
        time = np.arange(len(values)) / FPS
        stage_y, limit = (1.10, 1.16) if normalized else (710, 750)
        for index, subtask in enumerate(subtasks):
            left = (subtask["30FPSstart_frame"] - start) / FPS
            right = (subtask["30FPSend_frame"] + 1 - start) / FPS
            axis.axvspan(left, right, color="#edf1f6" if index % 2 == 0 else "#f8fafc", zorder=0)
            axis.text(
                (left + right) / 2,
                stage_y,
                textwrap.fill(subtask["name"], width=7),
                ha="center",
                va="center",
                fontsize=8,
                color="#475569",
            )
        axis.axhline(values[-1], color="#94a3b8", linestyle="--", linewidth=1, zorder=1)
        axis.plot(time, values, color="#2463eb", linewidth=2, zorder=3)
        axis.scatter(time[-1], values[-1], color="#2463eb", s=24, zorder=4)
        for event in summary["events"]:
            x = event["episode_seconds"]
            before = event["before"] if normalized else event["before_scaled"]
            after = event["after"] if normalized else event["after_scaled"]
            axis.axvline(x, ymax=0.9, color="#dc4b38", linestyle="--", linewidth=1.3, zorder=2)
            axis.scatter([x - 1 / FPS, x], [before, after], color="#dc4b38", s=26, zorder=5)
            ids = "、".join(event["reset_ids"])
            transition = f"{before:.2%} → {after:.2%}" if normalized else f"{before:.2f} → {after:.2f}"
            label = f"回滚任务 {ids}\n{transition}\n源帧 {event['frame']} · 原视频 {event['source_seconds']:.1f} s"
            axis.annotate(
                label,
                xy=(x, after),
                xytext=(x - 1.6, 0.14 if normalized else 90),
                ha="right",
                va="center",
                fontsize=10.5,
                color="#a53526",
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fff6f3", "edgecolor": "#f3c9bc"},
                arrowprops={"arrowstyle": "->", "color": "#dc4b38", "linewidth": 1.2},
                zorder=6,
            )
        axis.set_title(
            f"{summary['file']} / episode {summary['episode']}", loc="left", fontsize=14, color="#0f172a", pad=14
        )
        caption = f"源帧 {start}–{summary['end_frame']}  ·  "
        caption += f"最终进度 {progress[-1]:.0%}" if normalized else f"最终累计 {cumulative[-1]:.2f}"
        axis.set_title(caption, loc="right", fontsize=10.5, color="#64748b", pad=14)
        axis.set_xlim(0, 36)
        axis.set_ylim(0, limit)
        if normalized:
            axis.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        else:
            axis.set_yticks(np.arange(0, 701, 100))
        axis.set_xticks(np.arange(0, 37, 3))
        axis.tick_params(length=0, pad=7, labelbottom=True)
        axis.grid(axis="y", color="#dce3ed", linewidth=0.7)
        axis.set_ylabel("任务进度 p(t)" if normalized else "缩放后的累计贡献", labelpad=13)
    axes[-1].set_xlabel("episode 内时间（秒）", labelpad=13)
    fig.savefig(ROOT / f"recovery_{mode}.png", dpi=180, facecolor="white")
    fig.savefig(ROOT / f"recovery_{mode}.pdf", facecolor="white")
    plt.close(fig)

summaries = [result[0] for result in results]
(ROOT / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summaries, ensure_ascii=False, indent=2))
