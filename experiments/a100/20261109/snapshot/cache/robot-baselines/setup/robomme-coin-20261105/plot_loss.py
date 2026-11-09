import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

folder=Path(sys.argv[1])
rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
train=[r for r in rows if 'loss' in r]
val=[r for r in rows if 'val_flow_loss_32d' in r]
x=np.array([r['step'] for r in train]); y=np.array([r['loss'] for r in train])
vx=np.array([r['step'] for r in val]); vl=np.array([r['val_flow_loss_32d'] for r in val]); mae=np.array([r['val_action_normalized_mae_14d'] for r in val])
assert np.isfinite(y).all() and np.isfinite(vl).all() and np.isfinite(mae).all()
best=int(np.argmin(mae)); smooth=np.array([np.mean(y[max(0,i-10):i+1]) for i in range(len(y))])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#334155','text.color':'#0f172a','xtick.color':'#475569','ytick.color':'#475569','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(13.5,5.6),gridspec_kw={'width_ratios':[1.2,1]})
fig.patch.set_facecolor('#ffffff')
fig.suptitle('RoboMME fine-tuning | Three-coin task',x=.07,y=.97,ha='left',fontsize=19,fontweight='bold')
fig.text(.07,.91,'perceptual-framesamp-modul  •  80 train / 20 validation episodes  •  10,000 steps',fontsize=11,color='#64748b')
for ax in axes:
 ax.set_axisbelow(True);ax.grid(True,color='#e2e8f0',linewidth=.7);ax.set_xlim(0,10200)
 ax.set_xticks(np.arange(0,10001,2000));ax.xaxis.set_major_formatter(FuncFormatter(lambda n,p:f'{int(n):,}'))
 ax.set_xlabel('Training step');ax.axvline(vx[best],color='#64748b',ls='--',lw=1.2,alpha=.8)
a=axes[0]
a.plot(x,y,color='#93c5fd',lw=.8,alpha=.65,label='Train loss (logged minibatches)')
a.plot(x,smooth,color='#2563eb',lw=2,label='Train loss (11-point trailing mean)')
a.plot(vx,vl,color='#ea580c',lw=2,marker='o',ms=4,label='Validation flow loss')
a.set_yscale('log');a.set_ylim(.008,1.8);a.set_yticks([.01,.02,.05,.1,.2,.5,1]);a.yaxis.set_major_formatter(FuncFormatter(lambda n,p:f'{n:g}'))
a.set_ylabel('Flow-matching loss (log scale)');a.set_title('Training & validation loss',loc='left',fontsize=13,fontweight='bold',pad=12)
a.legend(loc='upper right',fontsize=8.5,frameon=True,facecolor='white',edgecolor='#e2e8f0')
b=axes[1];b.plot(vx,mae,color='#0d9488',lw=2.3,marker='o',ms=4)
b.scatter([vx[best]],[mae[best]],s=85,color='#0d9488',edgecolor='white',lw=1.5,zorder=4)
b.set_ylim(.075,.33);b.set_ylabel('Normalized action MAE (14 dimensions)');b.set_title('Held-out action prediction error',loc='left',fontsize=13,fontweight='bold',pad=12)
b.annotate(f'Best checkpoint: {vx[best]:,}\nMAE = {mae[best]:.5f}',xy=(vx[best],mae[best]),xytext=(4300,.19),fontsize=11,fontweight='bold',arrowprops={'arrowstyle':'->','color':'#0d9488','lw':1.2},color='#0f766e')
b.text(.04,.92,f'{100*(1-mae[best]/mae[0]):.1f}% lower than baseline',transform=b.transAxes,fontsize=12,color='#0f766e',fontweight='bold')
fig.text(.07,.055,'Dashed line: best checkpoint selected by validation action MAE. Training loss uses augmented minibatches.',fontsize=9,color='#64748b')
fig.text(.07,.025,'Validation: fixed 640 flow-loss samples and 80 action-rollout samples across 20 held-out episodes.',fontsize=9,color='#64748b')
fig.subplots_adjust(left=.07,right=.97,bottom=.18,top=.81,wspace=.28)
for suffix in ['png','pdf','svg']:fig.savefig(folder/f'robomme_training_loss.{suffix}',dpi=200,facecolor='white')
print(json.dumps({'train_points':len(train),'validation_points':len(val),'best_step':int(vx[best]),'best_mae':float(mae[best])}))
