"""Publish every natural attempt without mixing it into the old success rates."""
import sys,json,fcntl
from pathlib import Path
B=Path('/cache/robot-baselines');sys.path.insert(0,str(B/'setup/imprint-extra-20261105'))
import publish as shared
shared.OUT=B/'eval-results/imprint-failure-pilot-20261105';shared.DS=B/'viewer-datasets/IMPRINT-Natural-Failures-20261105';shared.TASKS=[t for t in shared.TASKS if t['task_id']=='S01']
shared.publish()
with (shared.DS/'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 p=shared.DS/'metadata.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()]
 for row in rows:
  model='full_attention' if 'full_attention' in row['sample_id'] else 'recurrent'
  x=json.loads((Path(row['evidence_path']).parents[2]/'result.json').read_text())
  row.update(trial_id=x.get('trial_id', x['requested_seed']),seed_selection=x.get('seed_selection','scene_sweep'),policy_seed=x['policy_seed'],failure_mode=x['failure_mode'],duplicate_failure=x['duplicate_failure'],collection='natural_failure_pilot',perturbation=None)
 tmp=p.with_suffix('.pilot.tmp')
 tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 tmp.replace(p)
