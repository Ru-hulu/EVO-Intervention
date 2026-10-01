"""试算两个异常 episode：原始 delta 累加、按 FailSet 回滚，不归一化。"""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
FPS = 30
SHORT_NAMES = {"1": "拿套筒", "2": "拿螺钉", "3": "对齐", "4": "插入", "5": "入盒"}


def calculate(stem, episode_id):
    annotation = json.loads((ROOT / "data" / f"{stem}.JSON").read_text())
    episode = next(
        e for task in annotation["tasks"] for e in task["episodes"] if e["episode"] == episode_id
    )
    subtasks = episode["subtasks"]
    rows = (ROOT / "data" / f"{stem}-delta.txt").read_text().splitlines()
    delta = np.array([float(row.split(maxsplit=1)[0]) for row in rows])
    start = subtasks[0]["30FPSstart_frame"]
    end = subtasks[-1]["30FPSend_frame"]
    progress = np.zeros(end - start + 1)
    contributions = {task["id"]: 0.0 for task in annotation["taskid"]}
    previous_remaining = set()
    finished = set()
    running = 0.0
    events = []
    next_frame = start

    for subtask in subtasks:
        now = subtask["nowid"]
        remaining = {value.strip() for value in subtask["restid"].split(",") if value.strip()}
        fail_set = (remaining | {now}) - previous_remaining
        removed = sum(contributions[task_id] for task_id in fail_set)
        first = subtask["30FPSstart_frame"]
        last = subtask["30FPSend_frame"]
        assert first == next_frame  # 这两个 episode 的标注连续，直接使用原始 30 fps 帧号。
        if removed > 0:
            events.append({
                "frame": first,
                "reset_ids": sorted(fail_set & finished),
                "removed_raw": removed,
                "before_raw": running,
                "after_raw": running - removed + float(delta[first]),
            })
        running -= removed
        finished.difference_update(fail_set)
        for task_id in fail_set:
            contributions[task_id] = 0.0

        for frame in range(first, last + 1):
            running += float(delta[frame])
            contributions[now] += float(delta[frame])
            progress[frame - start] = running
            assert np.isclose(running, sum(contributions.values()))

        finished.add(now)
        previous_remaining = remaining
        next_frame = last + 1

    final_raw = float(progress[-1])
    assert np.isfinite(progress).all()
    for event in events:
        event["source_seconds"] = event["frame"] / FPS
        event["episode_seconds"] = (event["frame"] - start) / FPS

    summary = {
        "file": stem,
        "episode": episode_id,
        "start_frame": start,
        "end_frame": end,
        "frame_count": len(progress),
        "final_raw": final_raw,
        "peak_raw": float(progress.max()),
        "events": events,
    }
    output = ROOT / f"{stem}-episode-{episode_id}-progress.txt"
    with output.open("w") as stream:
        stream.write("frame\tepisode_seconds\tdelta\traw_progress\tdescription\n")
        for offset, frame in enumerate(range(start, end + 1)):
            description = rows[frame].split(maxsplit=1)[1]
            stream.write(
                f"{frame}\t{offset / FPS:.6f}\t{delta[frame]:.6f}\t"
                f"{progress[offset]:.9f}\t{description}\n"
            )
    return summary, subtasks, progress


font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
plt.rcParams.update({
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
})
fig, axes = plt.subplots(2, 1, figsize=(13.5, 10), sharex=True, sharey=True)
fig.subplots_adjust(left=0.09, right=0.96, bottom=0.085, top=0.86, hspace=0.31)
fig.suptitle("两个异常 episode 的进度曲线", x=0.09, y=0.97, ha="left", fontsize=22, color="#0f172a")
fig.text(0.09, 0.928, "原始 delta 累加 → 按 FailSet 扣除贡献 → 继续累加（不归一化）", color="#475569", fontsize=12)
fig.text(0.09, 0.896, "灰色分区表示子任务；红线标出回滚起始帧；横轴为 episode 内经过的时间（30 fps）", color="#64748b", fontsize=10.5)
summaries = []

for axis, (stem, episode_id) in zip(axes, [("file-001", 5), ("file-005", 4)], strict=True):
    summary, subtasks, progress = calculate(stem, episode_id)
    summaries.append(summary)
    start = summary["start_frame"]
    time = np.arange(len(progress)) / FPS
    for index, subtask in enumerate(subtasks):
        left = (subtask["30FPSstart_frame"] - start) / FPS
        right = (subtask["30FPSend_frame"] + 1 - start) / FPS
        axis.axvspan(left, right, color="#edf1f6" if index % 2 == 0 else "#f8fafc", zorder=0)
        axis.text((left + right) / 2, 780, SHORT_NAMES[subtask["nowid"]], ha="center", va="center", fontsize=10, color="#475569")
    axis.plot(time, progress, color="#2463eb", linewidth=2, zorder=3)
    axis.scatter(time[-1], progress[-1], color="#2463eb", s=24, zorder=4)
    for event in summary["events"]:
        x = event["episode_seconds"]
        axis.axvline(x, ymax=0.92, color="#dc4b38", linestyle="--", linewidth=1.3, zorder=2)
        axis.scatter([x - 1 / FPS, x], [event["before_raw"], event["after_raw"]], color="#dc4b38", s=26, zorder=5)
        ids = "、".join(event["reset_ids"])
        label = (
            f"回滚任务 {ids}：扣除 {event['removed_raw']:.2f}\n"
            f"{event['before_raw']:.2f} → {event['after_raw']:.2f}（已计入本帧 delta）\n"
            f"源帧 {event['frame']} · 原视频 {event['source_seconds']:.1f} s"
        )
        axis.annotate(
            label, xy=(x, event["after_raw"]), xytext=(x - 1.6, 100),
            ha="right", va="center", fontsize=10.5, color="#a53526",
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "#fff6f3", "edgecolor": "#f3c9bc"},
            arrowprops={"arrowstyle": "->", "color": "#dc4b38", "linewidth": 1.2},
            zorder=6,
        )
    axis.set_title(f"{stem} / episode {episode_id}", loc="left", fontsize=14, color="#0f172a", pad=14)
    axis.set_title(
        f"源帧 {start}–{summary['end_frame']}  ·  峰值 {summary['peak_raw']:.2f}  ·  终值 {summary['final_raw']:.2f}",
        loc="right", fontsize=10.5, color="#64748b", pad=14,
    )
    axis.set_xlim(0, 36)
    axis.set_ylim(0, 820)
    axis.set_yticks(np.arange(0, 751, 150))
    axis.set_xticks(np.arange(0, 37, 3))
    axis.tick_params(length=0, pad=7, labelbottom=True)
    axis.grid(axis="y", color="#dce3ed", linewidth=0.7)
    axis.set_ylabel("delta 绝对累计值", labelpad=13)

axes[-1].set_xlabel("episode 内时间（秒）", labelpad=13)
fig.savefig(ROOT / "recovery_curves.png", dpi=180, facecolor="white")
fig.savefig(ROOT / "recovery_curves.pdf", facecolor="white")
plt.close(fig)
(ROOT / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
print(json.dumps(summaries, ensure_ascii=False, indent=2))
