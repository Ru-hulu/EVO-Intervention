"""Render two report episodes for slide 6, preserving every saved prediction.

Run: uv run --no-project --with matplotlib python render_curves.py
The bundled JSON records the source report, hashes, metrics and unchanged rows.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
FONT = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    family = "Arial Unicode MS"
else:
    family = "sans-serif"

plt.rcParams.update({
    "font.family": family, "font.size": 13, "svg.fonttype": "none",
    "path.simplify": False, "axes.unicode_minus": False,
    "figure.facecolor": "#fffefa", "axes.facecolor": "#fffefa",
    "savefig.facecolor": "#fffefa", "text.color": "#17263b",
    "axes.labelcolor": "#17263b", "xtick.color": "#626b77",
    "ytick.color": "#626b77", "svg.hashsalt": "evo-recovery-slide-6",
})

data = json.loads((ROOT / "selected_episodes.json").read_text())
for episode in data["episodes"]:
    rows = episode["frames"]
    time = [row["episode_time_seconds"] for row in rows]
    truth = [100 * row["ground_truth"]["progress"] for row in rows]
    prediction = [100 * row["prediction"]["progress"] for row in rows]
    assert len(rows) == episode["metrics"]["n"]
    assert abs(sum(abs(a-b) for a, b in zip(truth, prediction)) / len(rows)
               - episode["metrics"]["mae_pp"]) < 1e-9

    fig, ax = plt.subplots(figsize=(6.85, 3.95), dpi=180)
    fig.subplots_adjust(left=.12, right=.986, bottom=.18, top=.91)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dfe4ea", linewidth=.65)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#a8b5c2")
        ax.spines[side].set_linewidth(.75)
    ax.plot(time, truth, color="#2458a6", linewidth=2.1, label="真值", zorder=3)
    ax.plot(time, prediction, color="#d77829", linewidth=1.25, label="预测", zorder=4)
    for event in episode["events"]:
        t = event["time_seconds"]
        ax.axvline(t, color="#9a668d", linestyle=(0, (4, 3)), linewidth=1, zorder=2)
        ax.text(t, 1.02, f"{t:g} s", transform=ax.get_xaxis_transform(),
                color="#875778", fontsize=12, ha="center", va="bottom")
    ax.set_xlim(0, time[-1] + .3)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=12, length=3, width=.7)
    ax.set_xlabel("片段时间 / 秒", fontsize=13, labelpad=9)
    ax.set_ylabel("任务进度 / %", fontsize=13, labelpad=6)
    name = f"{episode['video']}-ep{episode['episode']:02d}"
    fig.savefig(ROOT / f"{name}.svg", metadata={"Date": None})
    fig.savefig(ROOT / f"{name}.png")
    plt.close(fig)
    print(name, len(rows), "unchanged frames")
