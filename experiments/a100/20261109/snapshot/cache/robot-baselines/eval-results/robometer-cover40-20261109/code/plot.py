"""Standalone diagnostic figure from actual prediction files."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
source = root.parent/'datasets/RMBench-CoverBlocks-40-20261105'
report = json.loads((root/'comparison.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for ax, sid, title in zip(axes[0], ['11-cover_blocks-memory-correction','12-cover_blocks-irreversible-failure'],
                         ['Correction, native success','Wrong uncover, irreversible failure']):
    pred = json.loads((root/'predictions'/(sid+'.json')).read_text())
    gt = json.loads((source/'progress'/(sid+'.json')).read_text())['dense_progress']
    fps = pred['fps']; points = pred['predictions']
    ax.plot(np.arange(len(gt))/fps, np.array(gt)*100, color='#147d73', label='Existing annotation')
    ax.plot([p['seconds'] for p in points], [p['predicted_progress']*100 for p in points],
            color='#b66416', marker='.', ms=3, lw=1, label='Robometer measured predictions')
    ax.set(title=title, xlabel='Video time (s)', ylabel='Progress (%)', ylim=(-2,102))
    ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.15)
episodes = report['episodes']
axes[1,0].bar(range(1,len(episodes)+1), [s['mae_1hz']*100 for s in episodes], color='#738b9c')
axes[1,0].set(xlabel='Trajectory index (see CSV)', ylabel='MAE (percentage points)', title='Per-trajectory error, regular 1 Hz samples')
for success, marker, color, label in [(True,'o','#147d73','30 correction successes'),(False,'x','#b66416','10 irreversible failures')]:
    group = [(i+1,s['final_prediction']*100) for i,s in enumerate(episodes) if s['success']==success]
    if group:
        x,y = zip(*group); axes[1,1].scatter(x,y,marker=marker,color=color,label=label)
axes[1,1].axhline(100,color='#147d73',alpha=.3,lw=1)
axes[1,1].axhline(0,color='#b66416',alpha=.3,lw=1)
axes[1,1].set(xlabel='Trajectory index',ylabel='Predicted terminal progress (%)',ylim=(-2,102),title='Terminal predictions (labels: success 100%, failure 0%)')
axes[1,1].legend(fontsize=8,frameon=False)
fig.suptitle(f'Zero-shot Robometer-4B | {len(episodes)}/40 trajectories | Causal 8-frame main-view prefix',fontsize=14)
fig.savefig(root/'comparison.png',dpi=180)
