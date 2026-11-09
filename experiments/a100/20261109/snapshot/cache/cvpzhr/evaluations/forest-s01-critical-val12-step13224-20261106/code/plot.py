"""Export validation prediction plots, using only evaluated observations."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'fa_success':'IMPRINT full attention: success',
       'fa_failure':'IMPRINT full attention: natural failure',
       'rec_failure':'IMPRINT recurrent: natural failure',
       'rec_success':'IMPRINT recurrent: success',
       'script_recovery':'Scripted recovery', 'script_failure':'Scripted failure'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir',type=Path)
    parser.add_argument('--model', choices=['fa','rec'])
    args=parser.parse_args();root=args.result_dir
    report=json.loads((root/'metrics.json').read_text())
    dataset=root/'viewer-data'/report['dataset']
    rows=[json.loads(l) for l in (dataset/'metadata.jsonl').read_text().splitlines()]
    if args.model: rows=[r for r in rows if r['category'].startswith(args.model+'_')]
    assert rows
    plots=root/'plots';plots.mkdir(exist_ok=True)
    combined,axes=plt.subplots(len(rows),1,figsize=(11,3.2*len(rows)),layout='constrained',squeeze=False)
    axes=axes.ravel()
    for i,(row,ax) in enumerate(zip(rows,axes),1):
        progress=json.loads((dataset/'progress'/f"{row['sample_id']}.json").read_text())
        fig,a=plt.subplots(figsize=(11,3.3),layout='constrained')
        for target in (a,ax):
            truth=progress['dense_progress'];fps=row['fps'];points=progress['prediction_points']
            target.plot([f/fps for f in range(len(truth))],[100*p for p in truth],color='#137e79',lw=2,label='Declining-progress label')
            target.plot([f/fps for f,p in points],[100*p for f,p in points],color='#df7b2b',lw=1.7,alpha=.9,label=f"FOREST step {report['step']}")
            target.set(title=f"{i}. {NAMES[row['category']]} | episode {row['episode_index']} | MAE {progress['metrics']['mae_pp']:.2f} pp",
                xlabel='Time (seconds)',ylabel='Progress (%)',ylim=(-3,103),xlim=(0,(len(truth)-1)/fps))
            target.grid(alpha=.18);target.legend(loc='upper left',fontsize=8)
        fig.savefig(plots/f"episode-{row['episode_index']:03d}-{row['category']}.png",dpi=160)
        plt.close(fig)
    combined.savefig(plots/(f'{args.model}-predictions.png' if args.model else 'all-predictions.png'),dpi=150)
    print(plots)

if __name__=='__main__':main()
