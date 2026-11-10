"""Exercise all plotting/data coverage gates with prior predictions in a QA directory."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from verify_and_plot import plot


def main():
    root=Path(__file__).resolve().parent
    policy=json.loads((root/'launch-policy.json').read_text());receipts={}
    for task,config_path in policy['tasks'].items():
        config=json.loads(Path(config_path).read_text())
        baseline=json.loads((root/'baseline_configs'/f'{task}.json').read_text())
        old=Path(baseline['output_dir']);grouped=defaultdict(list)
        rows=[json.loads(x) for x in (old/'evaluation/predictions.jsonl').read_text().splitlines()]
        for row in rows:grouped[row['episode_id']].append(row)
        episodes=[]
        for eid,group in grouped.items():
            a=np.array([r['predicted_pt'] for r in group]);b=np.array([r['target_pt'] for r in group])
            corr=None if np.ptp(a)==0 or np.ptp(b)==0 else float(np.corrcoef(a,b)[0,1])
            episodes.append(dict(episode_id=eid,samples=len(group),mae_pp=float(np.abs(a-b).mean()*100),pearson=corr))
        correlations=[r['pearson'] for r in episodes if r['pearson'] is not None]
        verified=dict(checkpoint=str(old/f"checkpoint-{baseline['expected_steps']}"),episode_metrics=episodes,
                      overall=dict(episodes=len(episodes),frames=len(rows),mae_pp=float(np.mean([abs(r['predicted_pt']-r['target_pt']) for r in rows])*100),
                                   pearson=float(np.mean(correlations)) if correlations else None))
        config['output_dir']=str(root/'plot-qa'/task);config['title']='绘图自检（旧预测）·'+config['title']
        (Path(config['output_dir'])/'evaluation').mkdir(parents=True,exist_ok=True)
        receipts[task]=plot(config,verified,grouped)
    (root/'plotting-verification.json').write_text(json.dumps(dict(status='passed',prior_predictions_only=True,tasks=receipts),indent=2)+'\n')
    print('Plotting preflight passed for every validation episode of all five tasks.')


if __name__=='__main__':main()
