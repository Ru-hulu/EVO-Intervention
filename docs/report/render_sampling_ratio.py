"""渲染数据集均衡统计图：各类样本的实际索引占比 vs 最终训练采样比例。

数据来源：
- 实际占比：扩充后正式索引 199,453 条（docs/report/report.html 的
  “各类样本总占比”表，来自远程 samples.jsonl）。
- 采样比例：qwen_training/dataloader.py 的 SAMPLING_PERCENTAGES。

用法：uv run --with matplotlib python docs/report/render_sampling_ratio.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent

# (输入 nowid, 输出 nowid): (索引样本数, 最终训练采样比例 %)
PAIRS = {
    (3, 1): (412, 3.14),
    (3, 2): (6422, 7.08),
    (1, 2): (8502, 8.44),
    (2, 1): (10052, 9.45),
    (1, 3): (10150, 9.52),
    (2, 3): (14496, 12.37),
    (3, 4): (20502, 10.00),
    (4, 4): (25647, 10.00),
    (1, 1): (28442, 10.00),
    (2, 2): (36233, 10.00),
    (3, 3): (38595, 10.00),
}

TOTAL = sum(count for count, _ in PAIRS.values())
CHANGE = "#2f7d6e"  # nowid 变化：跨阶段样本
SAME = "#2f5b9e"  # nowid 不变：同阶段与跨阶段不变样本

font_manager.fontManager.addfont("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
plt.rcParams.update({
    "font.family": "Arial Unicode MS", "font.size": 10.5,
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "savefig.facecolor": "#fffefa", "axes.facecolor": "#fffefa",
    "figure.facecolor": "#fffefa", "text.color": "#17263b",
    "axes.labelcolor": "#17263b", "xtick.color": "#626b77", "ytick.color": "#626b77",
})

items = sorted(PAIRS.items(), key=lambda kv: kv[1][0])
labels = [f"{a} \u2192 {b}" for (a, b), _ in items]
shares = [count / TOTAL * 100 for _, (count, _) in items]
ratios = [percent for _, (_, percent) in items]
counts = [count for _, (count, _) in items]
changed = [a != b for (a, b), _ in items]
colors = [CHANGE if flag else SAME for flag in changed]

fig = plt.figure(figsize=(11.4, 6.2), dpi=200)
ax = fig.add_axes([0.068, 0.245, 0.915, 0.585])
x = range(len(items))
width = 0.38

for i, (share, ratio) in enumerate(zip(shares, ratios)):
    ax.bar(i - width / 2, share, width, facecolor="none", edgecolor=colors[i],
           linewidth=1.2, alpha=0.55, zorder=3)
    ax.bar(i + width / 2, ratio, width, facecolor=colors[i], edgecolor=colors[i],
           linewidth=0.8, zorder=3)
    ax.text(i - width / 2 - 0.07, share + 0.45, f"{share:.2f}%", ha="center", va="bottom",
            fontsize=8.6, color="#5c6a7c", zorder=4)
    ax.text(i + width / 2, ratio + 0.45, f"{ratio:.2f}%", ha="center", va="bottom",
            fontsize=9.2, color=colors[i], fontweight="bold", zorder=4)

ax.axhline(10, color="#ad622c", linewidth=0.9, dashes=(5, 3), zorder=2)
ax.text(0.06, 10.4, "10% 基准：11 类中 5 类定为 10.00%", ha="left", va="bottom",
        fontsize=9.2, color="#ad622c", zorder=4)

ax.set_ylim(0, max(max(shares), max(ratios)) * 1.20)
ax.set_xlim(-0.7, len(items) - 0.3)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10.2)
for tick, color in zip(ax.get_xticklabels(), colors):
    tick.set_color(color)
ax.set_yticks([0, 5, 10, 15, 20])
ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0f}%")
ax.set_ylabel("占全部样本的比例 / 训练采样比例", fontsize=10.8, labelpad=8)
ax.set_xlabel("输入 nowid \u2192 输出 nowid（按索引样本数升序）", fontsize=10.8, labelpad=22)
ax.grid(axis="y", color="#e1e6ec", linewidth=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
for edge in ("left", "bottom"):
    ax.spines[edge].set_color("#aab7c6")
    ax.spines[edge].set_linewidth(0.65)
ax.tick_params(axis="both", length=0, pad=5)

for i, count in enumerate(counts):
    ax.text(i, -0.075, f"{count:,} 条", ha="center", va="top",
            fontsize=8.4, color="#8d99a8", transform=ax.get_xaxis_transform())

legend = [
    Patch(facecolor="none", edgecolor="#626b77", linewidth=1.2, label="索引实际占比（空心）"),
    Patch(facecolor="#626b77", edgecolor="#626b77", label="最终训练采样比例（实心）"),
    Line2D([], [], color=CHANGE, linewidth=6, label="nowid 变化（跨阶段）"),
    Line2D([], [], color=SAME, linewidth=6, label="nowid 不变"),
]
ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=9.6, ncol=2,
          handlelength=1.6, columnspacing=1.8, borderaxespad=0.2)

fig.text(0.068, 0.955, "数据集均衡：索引实际占比 vs 最终训练采样比例", fontsize=17.5,
         fontweight="bold", color="#17263b")
fig.text(0.068, 0.912,
         f"扩充后正式索引 {TOTAL:,} 条（示范 + 恢复）；跨阶段 nowid 变化类按原数量 k 再补 k 条独立随机窗口，"
         "索引共 35,268 → 70,536 条",
         fontsize=10.6, color="#44546a")
fig.text(0.068, 0.874,
         "训练按 nowid 变化 60.00% / nowid 不变 40.00% 分配配额，11 类中 5 类定为 10.00%，稀少的跨阶段类提高到 3.14%～12.37%",
         fontsize=10.6, color="#44546a")
fig.text(0.068, 0.092,
         "采样比例确定过程：先以 3 → 4 的 20,502 条为上限，对四类 nowid 不变样本作数量截断，得到 152,544 条的参考分布；",
         fontsize=9.0, color="#748197")
fig.text(0.068, 0.062,
         "再把其中五类约 13.44% 降至 10%，释放的约 17.20 个百分点平均分给其余六类，按 0.01 个百分点取整。",
         fontsize=9.0, color="#748197")
fig.text(0.068, 0.032,
         "数据来源：远程 samples.jsonl 全量索引统计（docs/report/report.html）与 qwen_training/dataloader.py 的 SAMPLING_PERCENTAGES。",
         fontsize=9.0, color="#748197")

for name in ("sampling-ratio-balance.png", "sampling-ratio-balance.svg"):
    fig.savefig(ROOT / name)
print("saved", ROOT / "sampling-ratio-balance.png", ROOT / "sampling-ratio-balance.svg")
