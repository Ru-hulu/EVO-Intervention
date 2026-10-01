"""绘制远程生成的 8 条进度曲线；在本目录旁读取 data/*.txt。"""

import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")
root = Path(__file__).resolve().parent
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

files = sorted((root / "data").glob("file-*-progress.txt"))
assert len(files) == 8
fig, axes = plt.subplots(4, 2, figsize=(13.2, 12.8), sharex=True, sharey=True)
fig.subplots_adjust(left=0.09, right=0.965, bottom=0.075, top=0.895, hspace=0.43, wspace=0.16)
fig.suptitle("任务进度曲线 · 8 个视频", x=0.09, y=0.973, ha="left", fontsize=21, color="#0f172a")
fig.text(0.09, 0.941, "逐帧进度 p(t)  |  原视频 30 fps  |  各图使用相同时间与进度刻度", fontsize=11.5, color="#64748b")
summary = []
for axis, path in zip(axes.flat, files, strict=True):
    rows = path.read_text(encoding="utf-8").splitlines()
    progress = np.array([float(line.split(maxsplit=1)[0]) for line in rows])
    assert np.isfinite(progress).all()
    assert (np.diff(progress) >= 0).all()
    assert progress[0] == 0
    assert progress[-1] == 1
    time = np.arange(len(progress)) / 30
    stem = path.name.removesuffix("-progress.txt")
    axis.plot(time, progress, color="#2463eb", linewidth=1.7, zorder=3)
    axis.fill_between(time, 0, progress, color="#2463eb", alpha=0.045, zorder=1)
    axis.scatter([0, time[-1]], [0, 1], color="#2463eb", s=15, zorder=4)
    axis.set_title(stem, loc="left", fontsize=13.5, color="#0f172a", pad=11)
    axis.set_title(f"{len(progress):,} 帧 · 末帧 {time[-1]:.2f} s", loc="right", fontsize=10.5, color="#64748b", pad=11)
    axis.set_xlim(0, 140)
    axis.set_ylim(-0.025, 1.045)
    axis.set_xticks([0, 30, 60, 90, 120])
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    axis.grid(color="#e7ecf2", linewidth=0.7)
    axis.tick_params(length=0, pad=7)
    summary.append(
        {
            "file": path.name,
            "frames": len(progress),
            "last_frame_seconds": float(time[-1]),
            "first_progress": float(progress[0]),
            "last_progress": float(progress[-1]),
        }
    )

fig.supxlabel("时间（秒）", y=0.028, fontsize=13, color="#334155")
fig.supylabel("任务进度 p(t)", x=0.027, fontsize=13, color="#334155")
fig.savefig(root / "progress_curves.png", dpi=190, facecolor="white")
fig.savefig(root / "progress_curves.pdf", facecolor="white")
plt.close(fig)
(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
with ZipFile(root / "progress_files.zip", "w") as archive:
    for path in files:
        archive.write(path, arcname=path.name)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("Saved", root / "progress_curves.png")
