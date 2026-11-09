"""Replot the four requested model/settings curves from the local report, unchanged."""
from pathlib import Path
import csv, json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=ROOT/'docs/report/bolt_progress_comparison/benchmark/results'
METHODS=[('D','RoboBrain 2.5 · 空白目标','#dd8a00','--'),
         ('E','RoboBrain 2.5 · 本段首尾','#c54832','-'),
         ('F','Dopamine Preview · 本段首尾','#00886b','-'),
         ('G','Dopamine Preview · 专家首尾','#238fc0','--')]
TASKS=[('robodojo__build_tower','搭积木'),('robodojo__classify_objects','物品分类')]
font=FontProperties(fname='/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':17,'svg.fonttype':'path',
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a8b5c4',
 'axes.labelcolor':'#526174','xtick.color':'#526174','ytick.color':'#526174',
 'axes.linewidth':1,'savefig.facecolor':'#fffefa'})
provenance={'source_report':str(ROOT/'docs/report/bolt_progress_comparison/report.html'),
 'series':[{'id':k,'label':label,'color':c,'style':ls} for k,label,c,ls in METHODS],
 'notes':['Raw signed start-to-current GRM scores, expressed in percent.',
 'No smoothing, clipping, forced endpoints, or interpolation.',
 'All selected scores lie between 0 and 100; displayed y-axis spans 0–100.',
 'Each task uses one query trajectory sampled at 3 FPS plus its final observation.',
 'Main-view observations; both wrist slots are white images.',
 'Dopamine model in this evaluation is Preview, not the Robo-Dopamine 2.0 model.',
 'Own endpoints include the evaluated trajectory final frame; expert endpoints use trajectory 2.',
 'No per-frame progress ground truth is included.'], 'tasks':[]}
for task,title in TASKS:
 fig,ax=plt.subplots(figsize=(8.4,5.5),facecolor='#fffefa')
 fig.subplots_adjust(left=.125,right=.98,top=.95,bottom=.16)
 ax.set_facecolor('#fffefa')
 records=[];sources=[]
 for k,label,color,ls in METHODS:
  path=SOURCE/'data'/task/f'{k}.jsonl'
  rows=[json.loads(x) for x in path.read_text().splitlines()]
  assert all(r['task_id']==task and r['method_id']==k for r in rows)
  assert all(r['status']=='ok' for r in rows)
  x=[r['timestamp_s'] for r in rows]; y=[r['score_percent'] for r in rows]
  assert all(0<=v<=100 for v in y)
  line,=ax.plot(x,y,color=color,ls=ls,lw=2.25,marker='o',ms=3,alpha=.96,label=label)
  line.set_gid(f'curve-{k}')
  records += [{'method':k,'timestamp_s':r['timestamp_s'],'score_percent':r['score_percent']} for r in rows]
  sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'points':len(rows)})
 ax.set_xlim(0,x[-1]);ax.set_ylim(-3,103)
 ax.set_yticks([0,25,50,75,100]);ax.set_xticks(list(range(0,int(x[-1])+1,5)))
 ax.grid(axis='both',alpha=1,color='#e4e9ef',linewidth=.9)
 ax.set_axisbelow(True);ax.tick_params(length=0,pad=9)
 ax.set_xlabel('时间（s）',fontproperties=font,fontsize=18,labelpad=11)
 ax.set_ylabel('GRM 预测分数（%）',fontproperties=font,fontsize=18,labelpad=10)
 fig.savefig(HERE/f'{task}.svg');fig.savefig(HERE/f'{task}.png',dpi=180)
 plt.close(fig)
 with (HERE/f'{task}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['method','timestamp_s','score_percent']);w.writeheader();w.writerows(records)
 provenance['tasks'].append({'task':task,'title':title,'sources':sources,'duration_s':x[-1]})
(HERE/'source.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
print('Rendered two tasks; four raw curves per task.')
