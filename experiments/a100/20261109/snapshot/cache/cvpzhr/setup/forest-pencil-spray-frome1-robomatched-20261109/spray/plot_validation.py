from pathlib import Path
import json,csv,math,zipfile,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator,MultipleLocator
ROOT=Path('/cache/cvpzhr/evaluations/forest-spray-frome1-e3-cleanval-20261109/curves')
ROOT.mkdir(parents=True,exist_ok=True)
def prepare_export():
 S=Path(__file__).resolve().parent;D=Path('/cache/cvpzhr/datasets/forest-spray-5fps-no-augmentation-20261109');E=ROOT.parent
 read=lambda p:json.loads(p.read_text());metrics=read(E/'metrics.json');archives=read(S/'checkpoint-verification.json')
 assert archives['status']=='passed' and [x['epoch'] for x in archives['archives']]==[1,2,3]
 assert metrics['coverage_complete'] and metrics['checkpoint_epoch']==3
 records={r['episode_id']:r for r in read(D/'plan.json')['records'] if r['split']=='validation'};assert len(records)==7
 lookup={ep['episode_id']:dict(ep,rows=[]) for ep in metrics['episodes']};seen=set()
 for rank in range(2):
  assert read(E/f'worker-{rank}.json')['phase']=='completed'
  for line in (E/f'predictions-rank{rank}.jsonl').read_text().splitlines():
   row=json.loads(line);key=(row['episode_id'],row['frame_index']);assert key not in seen;seen.add(key);lookup[row['episode_id']]['rows'].append(row)
 assert len(seen)==1919 and set(records)==set(lookup)
 for eid,ep in lookup.items():
  r=records[eid];ep.update(fps=r['fps'],source_frames=r['frames'],main_video=r['videos'][0]);ep['rows'].sort(key=lambda x:x['frame_index'])
  expected=list(range(0,r['frames'],6))
  if expected[-1]!=r['frames']-1:expected.append(r['frames']-1)
  assert [x['frame_index'] for x in ep['rows']]==expected
  with Path(r['progress_csv']).open() as f:labels={int(x['frame_index']):float(x['pt']) for x in csv.DictReader(f)}
  for row in ep['rows']:
   assert row['target_pt']==labels[row['frame_index']];row['time_seconds']=row['frame_index']/r['fps']
 result={'checkpoint':metrics['checkpoint'],'checkpoint_epoch':3,'checkpoint_step':archives['archives'][-1]['step'],'metrics':metrics['overall'],'source_metrics':str(E/'metrics.json'),'source_metrics_sha256':hashlib.sha256((E/'metrics.json').read_bytes()).hexdigest(),'validation_augmented':False,'sample_fps':5,'smoothing':False,'all_validation_episodes_included':True,'episodes':[lookup[k] for k in sorted(lookup)]}
 (ROOT/'validation_predictions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
prepare_export()
data=json.loads((ROOT/'validation_predictions.json').read_text());episodes=data['episodes'];metrics=data['metrics'];step=data['checkpoint_step']
font='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B9C4CF','axes.labelcolor':'#334155','xtick.color':'#475569','ytick.color':'#475569','text.color':'#172B40','figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'path','savefig.facecolor':'white'})
BLUE='#0072B2';ORANGE='#D55E00'
assert len(episodes)==7 and sum(len(e['rows']) for e in episodes)==1919
for ep in episodes:
 rows=ep['rows'];pred=np.array([x['predicted_pt'] for x in rows]);target=np.array([x['target_pt'] for x in rows]);time=np.array([x['time_seconds'] for x in rows])
 assert np.isfinite(pred).all() and np.isfinite(target).all() and np.all(np.diff(time)>0)
 assert math.isclose(np.abs(pred-target).mean(),ep['mae'],abs_tol=1e-12)
 assert math.isclose(np.corrcoef(pred,target)[0,1],ep['pearson'],abs_tol=1e-12)
assert math.isclose(sum(e['mae']*e['frames'] for e in episodes)/1919,metrics['mae'],abs_tol=1e-12)
assert math.isclose(sum(e['pearson'] for e in episodes)/7,metrics['pearson'],abs_tol=1e-12)

def panel(ax,ep):
 rows=ep['rows'];time=np.array([r['time_seconds'] for r in rows]);target=np.array([r['target_pt']*100 for r in rows]);pred=np.array([r['predicted_pt']*100 for r in rows])
 ax.plot(time,target,color=BLUE,lw=2.4,label='真值 PT',zorder=3)
 ax.plot(time,pred,color=ORANGE,lw=1.6,linestyle=(0,(4,2)),label='预测 PT',zorder=4)
 ax.fill_between(time,target,pred,color=ORANGE,alpha=.075,zorder=2,linewidth=0)
 ax.set_title(ep['episode_id'].split('/')[-1],loc='left',fontsize=14,pad=27)
 ax.text(0,1.035,f"MAE {ep['mae']*100:.2f} 个百分点     Pearson {ep['pearson']:.4f}",transform=ax.transAxes,ha='left',va='bottom',fontsize=10.5,color='#526477')
 ax.set_xlim(0,time[-1]);ax.set_ylim(-2,103);ax.yaxis.set_major_locator(MultipleLocator(25));ax.xaxis.set_major_locator(MaxNLocator(6,min_n_ticks=4))
 ax.grid(axis='y',color='#E4EAF0',linewidth=.8,zorder=0);ax.grid(axis='x',color='#EFF3F7',linewidth=.55,zorder=0)
 ax.set_xlabel('视频时间（秒）',labelpad=5);ax.set_ylabel('进度（%）',labelpad=7);ax.tick_params(length=3)
 return ax.get_legend_handles_labels()

fig,axes=plt.subplots(4,2,figsize=(15.5,14.8))
for ax,ep in zip(axes.flat,episodes):handles,labels=panel(ax,ep)
summary=axes.flat[-1];summary.axis('off')
summary.text(.02,.98,'完整验证集 · 7 条视频',ha='left',va='top',fontsize=19,color='#172B40',transform=summary.transAxes)
summary.text(.02,.76,f"整体 MAE：{metrics['mae_pp']:.2f} 个百分点\n平均 Pearson：{metrics['pearson']:.4f}",ha='left',va='top',fontsize=17,color=BLUE,linespacing=1.7,transform=summary.transAxes)
summary.text(.02,.34,f'Checkpoint：epoch 3 / step {step}\n观测点：1,919 个，完整覆盖\n训练 / 验证：均不添加人工静止或抖动\n原始进度回退保留；曲线未经平滑',ha='left',va='top',fontsize=11.5,color='#526477',linespacing=1.6,transform=summary.transAxes)
fig.subplots_adjust(left=.065,right=.983,top=.87,bottom=.065,hspace=.73,wspace=.22)
fig.text(.065,.978,'Forest · 喷瓶检查验证集进度曲线',fontsize=23,va='top')
fig.text(.065,.942,'全部 7 条验证视频   |   从 Count/Order/State epoch 1 继续微调三轮   |   5 FPS 原始预测   |   横轴按原视频时间对齐',fontsize=11,color='#53677A',va='top')
fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.985,.988),ncol=2,frameon=False,fontsize=12,handlelength=2.8)
fig.text(.065,.018,'蓝色实线：真实 PT；橙色虚线：预测 PT。MAE 按观测点计算；整体 Pearson 为 7 条视频相关系数的等权平均。',fontsize=10,color='#64748B')
fig.savefig(ROOT/'overview.png',dpi=180);fig.savefig(ROOT/'overview.svg');plt.close(fig)
for ep in episodes:
 fig,ax=plt.subplots(figsize=(9,4.8));handles,labels=panel(ax,ep)
 fig.subplots_adjust(left=.09,right=.98,top=.73,bottom=.16)
 fig.text(.09,.96,'喷瓶检查 · 真值与预测',fontsize=20,va='top')
 fig.text(.09,.877,f'Forest epoch 3 / step {step}   |   5 FPS   |   无人工干扰增强、无平滑',fontsize=10.5,color='#53677A',va='top')
 fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.99,.98),ncol=2,frameon=False,fontsize=11)
 name=ep['episode_id'].split('/')[-1];fig.savefig(ROOT/f'{name}.png',dpi=190);fig.savefig(ROOT/f'{name}.svg');plt.close(fig)
with (ROOT/'curves.csv').open('w',newline='') as f:
 cols=['episode_id','frame_index','time_seconds','target_pt','predicted_pt'];writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
 for ep in episodes:
  for row in ep['rows']:writer.writerow({key:row[key] for key in cols})
(ROOT/'metrics.json').write_text(json.dumps({'checkpoint':data['checkpoint'],'checkpoint_epoch':3,'checkpoint_step':step,'overall':metrics,'episodes':[{k:v for k,v in ep.items() if k!='rows'} for ep in episodes]},ensure_ascii=False,indent=2)+'\n')
manifest={'status':'passed','episodes':7,'samples':1919,'all_validation_episodes_included':True,'metrics_recomputed_from_plotted_rows':True,'smoothing':False,'ground_truth_verified_against_original_pt':True,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.suffix in ['.png','.svg','.json','.csv'] and p.name!='manifest.json'}}
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(ROOT/'forest-spray-validation-curves.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(ROOT.iterdir()):
  if p.suffix in ['.png','.svg','.json','.csv']:z.write(p,arcname=p.name)
print(json.dumps({'output_dir':str(ROOT),'episodes':7,'samples':1919,'mae_pp':metrics['mae_pp'],'pearson':metrics['pearson']},ensure_ascii=False))
