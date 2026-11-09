"""Render slide 8's measured-label curve and a same-duration linear-time baseline."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / 'rollback-example.tsv').open(), delimiter='\t'))
t = np.array([float(r['episode_seconds']) for r in rows])
p = np.array([float(r['progress']) for r in rows])
# Analytic baseline only: elapsed time normalized by this episode's duration.
# The upper curve preserves all existing per-frame labels without smoothing.
p_time = (t - t[0]) / (t[-1] - t[0])
font_manager.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
plt.rcParams.update({
    'font.family': 'Arial Unicode MS', 'font.size': 10.5,
    'axes.unicode_minus': False, 'svg.fonttype': 'path',
    'path.simplify': False, 'savefig.facecolor': '#fffefa',
    'axes.facecolor': '#fffefa', 'figure.facecolor': '#fffefa',
    'text.color': '#17263b', 'axes.labelcolor': '#17263b',
    'xtick.color': '#626b77', 'ytick.color': '#626b77',
})
fig = plt.figure(figsize=(9, 3.85), dpi=200)
upper = fig.add_axes([.105, .46, .875, .405])
lower = fig.add_axes([.105, .115, .875, .20], sharex=upper)
for ax in (upper, lower):
    ax.set_xlim(0, 37)
    ax.set_ylim(0, 1.12)
    ax.spines[['top', 'right']].set_visible(False)
    for edge in ('left', 'bottom'):
        ax.spines[edge].set_color('#aab7c6')
        ax.spines[edge].set_linewidth(.65)
    ax.tick_params(axis='both', length=0, pad=4, labelsize=10.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis='y', color='#e1e6ec', linewidth=.6)
    ax.set_axisbelow(True)
    ax.axvline(21.6, color='#ad622c', linewidth=.9, dashes=(4, 3), zorder=2)
upper.set_yticks([0, .25, .5, .75, 1])
lower.set_yticks([0, .5, 1])
lower.set_xticks([0, 8, 16, 24, 32])
upper.tick_params(axis='x', labelbottom=False)
upper.set_ylabel('进度 p(t)', fontsize=11, labelpad=7)
lower.set_ylabel('进度', fontsize=11, labelpad=19)
lower.set_xlabel('episode 内时间（秒）', fontsize=10.8, labelpad=6)

stage_id = {'拿起套筒': 1, '拿起螺钉': 2, '将螺钉和套筒对齐并插入': 3, '将装配好的组件放入盒子': 4}
colors = {1: '#e6f0f5', 2: '#fff0e2', 3: '#e8f2e4', 4: '#f9e8e9'}
starts = [0] + [i for i in range(1, len(rows)) if rows[i]['description'] != rows[i-1]['description']]
for n, start in enumerate(starts):
    end = starts[n+1] if n+1 < len(starts) else len(rows)-1
    left, right = t[start], t[end]
    sid = stage_id[rows[start]['description']]
    upper.axvspan(left, right, color=colors[sid], linewidth=0, zorder=0)
    upper.text((left+right)/2, 1.057, str(sid), ha='center', va='center', fontsize=10.2, color='#626b77')
upper.axhline(1, color='#a9bfdc', linewidth=.65, linestyle=(0, (2, 2)))
upper.plot(t, p, color='#2458a6', linewidth=1.55, solid_capstyle='round', zorder=3)
drop = int(np.flatnonzero(np.diff(p) < -.1)[0]) + 1
upper.scatter(t[[drop-1, drop]], p[[drop-1, drop]], color='#ad622c', s=16, zorder=4)
upper.annotate(f'{p[drop-1]:.1%} → {p[drop]:.1%}', xy=(t[drop-1], p[drop-1]), xytext=(24.1, .92),
               color='#ad622c', fontsize=10.5, ha='left', va='center',
               arrowprops={'arrowstyle':'-', 'lw':.75, 'color':'#ad622c'})
upper.text(.0, 1.14, '视觉变化分配 + 回退扣减', transform=upper.transAxes,
           fontsize=11.6, fontweight='bold', va='bottom', color='#2458a6')
upper.text(1, 1.14, '21.6 s：重新拿起螺钉', transform=upper.transAxes,
           fontsize=10.3, ha='right', va='bottom', color='#ad622c')

lower.plot(t, p_time, color='#778697', linewidth=1.55, solid_capstyle='round', zorder=3)
lower.text(0, 1.17, '仅按时间线性标注：p(t) = t/T', transform=lower.transAxes,
           fontsize=11.6, fontweight='bold', va='bottom', color='#626b77')
lower.text(1, 1.17, '单调递增，无法体现回退', transform=lower.transAxes,
           fontsize=10.3, ha='right', va='bottom', color='#626b77')

for ext in ('svg', 'png'):
    fig.savefig(ROOT / f'rollback-time-comparison.{ext}', dpi=200, metadata={'Creator': 'Matplotlib'})
plt.close(fig)
print(f'Saved SVG/PNG: {len(rows)} unchanged label samples; linear-time baseline 0–100%; duration {t[-1]:.6f} s.')
