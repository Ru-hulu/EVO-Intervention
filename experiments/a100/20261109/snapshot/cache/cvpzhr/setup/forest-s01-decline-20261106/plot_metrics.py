"""Plot logged training loss separately from full-validation progress errors."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
src=Path('/cache/cvpzhr/outputs/forest-s01-decline-finetune-20261106')
migrated=src.with_name(src.name+'-4gpu')
if (migrated/'train.jsonl').exists(): src=migrated
out=Path('/cache/cvpzhr/evaluations/forest-s01-val5-20261106')
t=[json.loads(l) for l in (src/'train.jsonl').read_text().splitlines()]
v=[json.loads(l) for l in (src/'validation.jsonl').read_text().splitlines()]
x=np.array([r['step'] for r in t]);y=np.array([r['loss'] for r in t])
fig,axes=plt.subplots(2,1,figsize=(11,7.6),layout='constrained')
a=axes[0];a.plot(x,y,color='#137e79',alpha=.20,lw=.9,label='Logged minibatch loss (every 10 updates)')
sm=[y[max(0,i-19):i+1].mean() for i in range(len(y))]
a.plot(x,sm,color='#137e79',lw=2,label='20-record moving average (~200 updates)')
a.set(title='FOREST S01 fine-tuning — training loss',ylabel='Soft-label cross-entropy');a.legend(fontsize=9);a.grid(alpha=.18)
a=axes[1]
for key,label,color in [('mae_pp','Full-val MAE','#137e79'),('rmse_pp','Full-val RMSE','#d97326')]:
 a.plot([r['step'] for r in v],[r[key] for r in v],'o-',label=label,color=color,lw=2)
 for r in v:a.annotate(f"{r[key]:.2f}",(r['step'],r[key]),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
a.set(xlabel='Optimizer update',ylabel='Progress error (percentage points)',title='Full validation (28 trajectories) — metrics, not training loss',ylim=(0,10.5));a.legend(fontsize=9);a.grid(alpha=.18)
for a in axes:a.set_xlim(0,max(x)*1.025)
gpu_label='GPU 0 + 1 + 2 + 3' if src==migrated else 'GPU 2 + 3'
fig.suptitle(f'Latest logged step: {x[-1]:,} / 13,224 | 3 epochs | {gpu_label}',fontsize=12)
fig.savefig(out/'loss.png',dpi=170)
fig.savefig(out/'loss.pdf')
print(json.dumps(dict(step=int(x[-1]),recent_loss=float(y[-10:].mean()),seconds=float(np.mean([r['step_seconds'] for r in t[-10:]])),validation=v),indent=2))
