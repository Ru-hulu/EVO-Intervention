import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from history_probe import ROOT
r=[json.loads(x) for x in (ROOT/'results.jsonl').read_text().splitlines()]
c=[json.loads(x) for x in (ROOT/'compression-results.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
for cond,label in [('original','Original history'),('swapped','Matched success history'),('empty','Empty history')]:
 rr=[x for x in r if x['failure_episode']==55 and x['direction']=='failure_view' and x['condition']==cond]
 axes[0].plot([x['failure_frame']/20 for x in rr],[x['predicted_pt']*100 for x in rr],'o-',label=label)
axes[0].axvline(31.3,ls=':',color='gray');axes[0].set(title='Same ep55 image, different history',xlabel='Current time (s)',ylabel='Predicted progress (%)',ylim=(-3,103));axes[0].legend()
names=['original','protect_failure','protect_neutral','raw_failure_anchors','raw_neutral_anchors','reverse_history_tokens']
labels=['Original','Protect failure','Protect neutral','Raw failure frames','Raw neutral frames','Reverse tokens']
values=[next(x['predicted_pt']*100 for x in c if x['current_frame']==900 and x['condition']==n) for n in names]
axes[1].barh(labels,values,color=['#566b80','#dd7b32','#a3acb8','#148679','#9bc7c1','#8b74ac']);axes[1].invert_yaxis()
for i,v in enumerate(values):axes[1].text(v+1,i,f'{v:.2f}%',va='center',fontsize=9)
axes[1].set(title='45 s: GT = 0; no intervention fixes the error',xlabel='Predicted progress (%)',xlim=(0,114))
fig.suptitle('FOREST step14470 | History ablation (diagnostic, batch=1)',fontsize=14)
fig.savefig(ROOT/'history-ablation.png',dpi=160)
