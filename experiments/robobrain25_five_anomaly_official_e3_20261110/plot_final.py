"""Verified unsmoothed full validation curves, CSV and archive."""
import argparse,csv,hashlib,math,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from common import *
from metrics import correlation

def main(c):
 d=Path(c['output_dir'])/'evaluation';m=read(d/'metrics.json');assert m['coverage_complete'] and m['checkpoint_epoch']==3
 rr=rows(d/'predictions.jsonl');source=read(Path(c['data_dir'])/'episodes.json');groups=defaultdict(list)
 assert len(rr)==c['counts']['validation']['samples']
 for row in rr:groups[row['episode_id']].append(row)
 assert len(groups)==c['counts']['validation']['episodes']
 for eid,entries in groups.items():
  with Path(source[eid]['original_progress_csv']).open() as f:labels={int(x['frame_index']):float(x['pt']) for x in csv.DictReader(f)}
  for row in entries:assert row['target_pt']==labels[row['source_frame_index']]
  entries.sort(key=lambda x:x['source_frame_index'])
  assert all(a['source_frame_index']<b['source_frame_index'] for a,b in zip(entries,entries[1:]))
 mae=100*np.mean([abs(x['predicted_pt']-x['target_pt']) for x in rr]);assert abs(mae-m['overall']['mae_pp'])<1e-9
 corrs=[correlation([x['predicted_pt'] for x in r],[x['target_pt'] for x in r]) for r in groups.values()];defined=[x for x in corrs if x is not None]
 if defined:assert abs(np.mean(defined)-m['overall']['pearson_episode_mean'])<1e-10
 p=d/'curves';p.mkdir(exist_ok=True)
 font='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc';font_manager.fontManager.addfont(font)
 plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 def panel(ax,eid,records):
  t=np.array([x['timestamp_seconds'] for x in records]);y=np.array([x['target_pt']*100 for x in records]);z=np.array([x['predicted_pt']*100 for x in records])
  r=correlation(z,y);label=f'{r:.4f}' if r is not None else '未定义'
  ax.plot(t,y,color='#0072B2',lw=2,label='真实 PT');ax.plot(t,z,color='#D55E00',lw=1.5,label='预测 PT')
  ax.set_title(f"{eid.split('/')[-1]}    MAE {np.mean(abs(y-z)):.2f} pp    Pearson {label}",loc='left',fontsize=11,pad=10)
  ax.set_xlabel('视频时间（秒）');ax.set_ylabel('进度（%）');ax.set_xlim(0,t[-1]);ax.set_ylim(min(-2,float(z.min())-2),max(103,float(z.max())+2));ax.grid(alpha=.18)
  return ax.get_legend_handles_labels()
 n=len(groups);fig=plt.figure(figsize=(15,14));grid=fig.add_gridspec(4,2)
 for i,(eid,entries) in enumerate(sorted(groups.items())):handles,labels=panel(fig.add_subplot(grid[i//2,i%2]),eid,entries)
 summary=fig.add_subplot(grid[3,1] if n==7 else grid[3,:]);summary.axis('off')
 pearson=m['overall']['pearson_episode_mean'];summary.text(.02,.88,f"完整验证集：{n} 条／{len(rr):,} 个观测\nMAE：{mae:.2f} 个百分点\n平均 Pearson：{pearson:.4f}" if pearson is not None else f'完整验证集 {n} 条；Pearson 未定义',va='top',fontsize=17,linespacing=1.7)
 fig.suptitle(f"RoboBrain2.5 · {NAMES[c['task']]} · 验证集进度预测",fontsize=22,y=.985)
 fig.text(.08,.947,f"官方原始权重起训 3 epoch · checkpoint-{c['total_steps']} · 无人工增强 · 使用同轨迹真实末帧参考",fontsize=12,color='#526477')
 fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.98,.947),ncol=2,frameon=False)
 fig.subplots_adjust(left=.08,right=.98,top=.89,bottom=.065,hspace=.58,wspace=.22)
 fig.text(.08,.025,'蓝色：真实进度；橙色：模型预测。曲线未经平滑。MAE 按帧等权，Pearson 按轨迹等权。',fontsize=11,color='#526477')
 fig.savefig(p/'overview.png',dpi=160);fig.savefig(p/'overview.pdf');plt.close(fig)
 for eid,entries in sorted(groups.items()):
  fig,ax=plt.subplots(figsize=(10,4.8));panel(ax,eid,entries);ax.legend();fig.tight_layout();fig.savefig(p/(eid.split('/')[-1]+'.png'),dpi=180);plt.close(fig)
 with (p/'curves.csv').open('w',newline='') as f:
  keys=['episode_id','source_frame_index','timestamp_seconds','target_pt','predicted_pt'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for row in rr:w.writerow({k:row[k] for k in keys})
 save(p/'metrics.json',m)
 save(p/'manifest.json',dict(status='passed',episodes=n,samples=len(rr),checkpoint=m['checkpoint'],source_PT_verified=True,metrics_recomputed=True,smoothing=False,
    files={f.name:sha(f) for f in p.iterdir() if f.is_file() and f.suffix in ['.png','.pdf','.csv','.json'] and f.name!='manifest.json'}))
 with zipfile.ZipFile(p/'validation_curves.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(p.iterdir()):
   if f.suffix in ['.png','.pdf','.csv','.json']:z.write(f,f.name)
 print('CURVES_READY',p,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args();main(read(a.config))
