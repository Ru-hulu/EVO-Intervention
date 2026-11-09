"""Plot a read-only snapshot; retain source logs and full artifacts on A100."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

root=Path('/cache/cvpzhr/outputs/forest-crossattn-v2-e3-20261101-210436')
source=root/'train.jsonl'
raw=source.read_bytes()
complete=raw[:raw.rfind(b'\n')+1]
rows=[json.loads(line) for line in complete.splitlines() if line.strip()]
steps=np.array([r['step'] for r in rows]);loss=np.array([r['loss'] for r in rows])
assert np.isfinite(loss).all() and (np.diff(steps)>0).all()
per_epoch=10537;total=31611;window=50;last=int(steps[-1])
trend=np.convolve(loss,np.ones(window)/window,mode='valid')
out=root/'curves'/'training-loss';out.mkdir(parents=True,exist_ok=True)
stem=f'CrossAttention-v2-训练Loss-step{last}'
font='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),
    'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'savefig.facecolor':'white'})
fig,ax=plt.subplots(figsize=(14,7.5))
fig.subplots_adjust(left=.075,right=.975,top=.76,bottom=.19)
fig.text(.075,.95,'Forest + Cross-Attention v2 · 训练 Loss',fontsize=23,color='#172033',va='top')
fig.text(.075,.885,f'截至 {last:,} / {total:,} 步 · {last/per_epoch:.2f} / 3 epoch · 3 × A100 · batch=4/卡（全局12）',
         fontsize=12,color='#64748B',va='top')
ax.plot(steps,loss,color='#A4BFE4',alpha=.65,lw=.65,label='单步 Loss（每 10 步记录）')
ax.plot(steps[window-1:],trend,color='#1758AD',lw=2.3,label='最近 50 个日志点移动平均')
for boundary in (per_epoch,2*per_epoch):
    ax.axvline(boundary,color='#94A3B8',ls=(0,(4,4)),lw=1,zorder=0)
ax.axvspan(last,total,color='#F1F5F9',zorder=0)
ymax=max(5,float(np.ceil(loss.max()*1.06)))
if total-last>500:
    ax.text((last+total)/2,ymax*.4,'快照之后\n尚无记录',ha='center',va='center',fontsize=11,color='#8391A5')
ax.scatter([last],[trend[-1]],color='#1758AD',s=25,zorder=4)
ax.annotate(f'近期均值 {trend[-1]:.3f}',xy=(last,trend[-1]),xytext=(-145,44),textcoords='offset points',
            fontsize=11,color='#1758AD',arrowprops={'arrowstyle':'-','color':'#1758AD','lw':.8},
            bbox={'boxstyle':'round,pad=.3','facecolor':'white','edgecolor':'none','alpha':.95})
ax.set(xlim=(0,total),ylim=(0,ymax),xlabel='优化器更新步数（step）',ylabel='训练 Loss（软标签交叉熵）')
ax.set_xticks([0,5000,10000,15000,20000,25000,30000])
ax.grid(axis='y',color='#E2E8F0',lw=.8);ax.set_axisbelow(True)
secondary=ax.secondary_xaxis('top',functions=(lambda x:x/per_epoch,lambda x:x*per_epoch))
secondary.set_xticks([0,1,2,3],labels=['开始','第 1 轮结束','第 2 轮结束','第 3 轮结束'])
secondary.tick_params(length=0,pad=9,colors='#64748B')
ax.legend(loc='upper right',frameon=False,fontsize=11)
fig.text(.075,.105,'浅蓝保留全部日志点；深蓝为最近 50 个日志点均值（约 500 步），仅用于观察趋势。',color='#64748B',fontsize=11)
fig.text(.075,.055,'训练仍在进行；完整验证将在第 3 轮结束后执行。训练 Loss 不等同于进度误差百分点。',color='#64748B',fontsize=11)
for extension in ('png','pdf'):
    fig.savefig(out/f'{stem}.{extension}',dpi=160)
plt.close(fig)
(out/f'{stem}.jsonl').write_bytes(complete)
summary={'last_step':last,'total_steps':total,'epoch':last/per_epoch,'logged_points':len(rows),
    'first_logged_loss':float(loss[0]),'last_logged_loss':float(loss[-1]),
    'first_50_logged_mean':float(loss[:window].mean()),'last_50_logged_mean':float(trend[-1]),
    'window_logged_points':window,'source_snapshot_sha256':hashlib.sha256(complete).hexdigest(),
    'png':str(out/f'{stem}.png'),'pdf':str(out/f'{stem}.pdf')}
(out/f'{stem}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
