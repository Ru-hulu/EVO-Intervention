"""Publication-sized curves and combined plate from audited remote predictions."""
from pathlib import Path
import sys
import json,csv,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader,PdfWriter,Transformation,PageObject
from PIL import Image

ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1];DATA=ROOT/'source/results';PDF=ROOT/'output/pdf'
MODELS=['robometer-4b','robobrain25-4b','dopamine-3b','procvlm-2b','r2vlm-alfred-rl','vlac-2b']
NAMES=['Robometer-4B','RoboBrain2.5-4B*','Robo-Dopamine-3B*','ProcVLM-2B',r'R$^2$VLM-Alfred-RL','VLAC-2B']
COLORS=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#6F57A5']
STYLES=['solid',(0,(4,1.8)),'dashdot','solid',(0,(2.4,1.3)),(0,(1,1))]
proof=json.loads((DATA/'completion.json').read_text());assert proof['status']=='passed'
manifest=json.loads((DATA/'manifest.json').read_text());row=manifest['row'];n=len(row['frame_indices']);assert n==289
records={m:json.loads((DATA/'predictions'/f'{m}.json').read_text()) for m in MODELS}
manifest_sha=hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest()
for model,r in records.items():
 assert r['model']==model and r['key']==row['key'] and r['manifest_sha256']==manifest_sha
 for key in ['target_pt','frame_indices','timestamps','progress_sha256']:assert r[key]==row[key],key
 assert len(r['predictions'])==len(r['raw_responses'])==n
 assert r['uses_future_endpoint_reference']==(model in ['robobrain25-4b','dopamine-3b'])
t=np.asarray(row['timestamps']);truth=np.asarray(row['target_pt'])
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':6.7,'axes.labelsize':7.2,
 'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,'pdf.fonttype':42,
 'axes.linewidth':.45,'xtick.major.width':.45,'ytick.major.width':.45,'xtick.major.size':2.2,'ytick.major.size':2.2})
width=16.8/2.54;height=3.4/2.54
fig=plt.figure(figsize=(width,height),facecolor='white')
# Plot endpoints align with the centers of the first/last image panels.
strip=PdfReader(PDF/'O18-episode-000028-uniform-6-frames.pdf').pages[0]
W=float(strip.mediabox.width);panel=(W-5*1.5)/6;left=(panel/2)/W;right=1-left
ax=fig.add_axes([left,.27,right-left,.49])
for model,name,color,style in zip(MODELS,NAMES,COLORS,STYLES):
 y=np.array([np.nan if x is None else x for x in records[model]['predictions']],dtype=float)
 ax.plot(t,y,label=name,color=color,linewidth=.9,linestyle=style,alpha=.96,zorder=3)
ax.plot(t,truth,label='Visual PT (reference)',color='#202020',linewidth=1.35,linestyle=(0,(5,2.2)),zorder=5)
selected=json.loads((ROOT/'source/figure-verification.json').read_text())['selected_timestamps_seconds']
for x in selected[1:-1]:ax.axvline(x,color='#D8D8D8',linewidth=.4,linestyle=(0,(2,3)),zorder=0)
all_y=np.array([v for r in records.values() for v in r['predictions'] if v is not None],dtype=float)
lo=min(0.,float(all_y.min()));hi=max(1.,float(all_y.max()));ax.set_ylim(lo-.025*(hi-lo),hi+.025*(hi-lo))
ax.set_xlim(t[0],t[-1])
ticks=np.arange(np.ceil(lo*2)/2,np.floor(hi*2)/2+.25,.5)
ax.set_yticks(ticks);ax.set_yticklabels([f'{v:g}' for v in ticks])
ax.set_xticks(selected);ax.set_xticklabels([f'{v:.1f}' for v in selected])
ax.set_xlabel('Time (s)',labelpad=2);ax.set_ylabel('Progress',labelpad=4)
ax.grid(axis='y',color='#E5E5E5',linewidth=.4);ax.set_axisbelow(True)
ax.spines[['top','right']].set_visible(False)
handles,labels=ax.get_legend_handles_labels();order=[6,3,0,4,1,5,2]
# Matplotlib arranges legends by columns; this order gives two balanced rows.
fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='upper center',
 bbox_to_anchor=(.5,1.015),ncol=4,frameon=False,handlelength=2.7,columnspacing=1.2,
 labelspacing=.3,handletextpad=.5,borderaxespad=0)
fig.text(.5,.015,'* Uses the true final frame of this trajectory.',ha='center',va='bottom',fontsize=5.5,color='#444444')
curve=PDF/'O18-episode-000028-six-model-curves.pdf'
fig.savefig(curve);fig.savefig(ROOT/'O18-episode-000028-six-model-curves.png',dpi=450);plt.close(fig)
page=PdfReader(curve).pages[0];assert abs(float(page.mediabox.width)-W)<.02
GAP=3.4;H=float(page.mediabox.height)+float(strip.mediabox.height)+GAP
combined=PageObject.create_blank_page(width=W,height=H)
combined.merge_page(page)
combined.merge_transformed_page(strip,Transformation().translate(0,float(page.mediabox.height)+GAP))
writer=PdfWriter();writer.add_page(combined);writer.add_metadata({'/Title':'O18 validation episode 000028: observations and six official model predictions'})
with (PDF/'O18-episode-000028-combined.pdf').open('wb') as f:writer.write(f)
with (ROOT/'O18-episode-000028-predictions.csv').open('w',newline='') as f:
 out=csv.writer(f);out.writerow(['source_frame','time_seconds','visual_pt',*MODELS])
 for i in range(n):out.writerow([row['frame_indices'][i],t[i],truth[i],*[records[m]['predictions'][i] for m in MODELS]])
# Keep the image strip and the vector plot exactly equal in physical width.
(ROOT/'figure.tex').write_text('''% Requires \\usepackage{graphicx}; copy the combined PDF beside your manuscript.
\\begin{figure}[t]
  \\centering
  \\includegraphics[width=\\linewidth]{O18-episode-000028-combined.pdf}
  \\caption{Qualitative results on one held-out O18 trajectory (episode 000028). Top: six uniformly sampled main-camera observations. Bottom: progress predictions from six official models at 5 FPS and the reference PT computed by cumulative normalized visual differences over all original frames. Predictions retain their native scales, without clipping or smoothing. RoboBrain2.5 and Robo-Dopamine use the true final frame as a reference. This additional trajectory is evaluated separately from the frozen Group 1 benchmark.}
  \\label{fig:o18-qualitative}
\\end{figure}
''')
verification=dict(status='passed',observations_per_model=n,models=MODELS,source_frame_count=row['frames'],
 no_smoothing=True,no_invalid_value_imputation=True,common_width_cm=W/72*2.54,combined_height_cm=H/72*2.54,
 invalid_counts={m:sum(x is None for x in records[m]['predictions']) for m in MODELS},
 same_frame_times_and_targets_verified=True,manifest_sha256=manifest_sha,include_in_group1_results=False)
(ROOT/'source/curves-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
