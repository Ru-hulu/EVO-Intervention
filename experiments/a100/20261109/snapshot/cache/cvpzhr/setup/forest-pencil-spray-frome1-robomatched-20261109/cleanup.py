from pathlib import Path
import json,shutil,time,subprocess,hashlib
ROOT=Path('/cache/cvpzhr/setup/forest-pencil-spray-frome1-robomatched-20261109')
assert json.loads((ROOT/'preflight.json').read_text())['status']=='passed'
policy=json.loads((ROOT/'launch-policy.json').read_text())
removed=[];bytes_total=0
procs=subprocess.check_output(['ps','-eo','pid,args'],text=True)
for task in ['pencil','spray']:
 old=Path(f'/cache/cvpzhr/setup/forest-{task}-noaug-e3-20261109')
 state=json.loads((old/'status.json').read_text());assert state['phase']=='completed'
 # Check exact old entry points and output arguments; no other jobs are touched.
 for line in procs.splitlines():
  assert not (str(old) in line and any(x in line for x in ['pipeline.py','launch.py','train_stage3','evaluate_final.py'])),line
 targets=[Path(f'/cache/cvpzhr/outputs/forest-{task}-frombase-noaug-e3-20261109'),Path(f'/cache/cvpzhr/evaluations/forest-{task}-e3-noaug-val-20261109')]
 if (old/'benchmarks').exists():targets.append(old/'benchmarks')
 for p in targets:
  assert p.is_dir() and not p.is_symlink()
  files=[f for f in p.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in files)
  removed.append({'path':str(p),'files':len(files),'bytes':size});bytes_total+=size;shutil.rmtree(p);assert not p.exists()
 for name in ['train.log','evaluation.log','evaluation-aggregate.log','plotting.log','checkpoint-verification.log','checkpoint-verification.json','benchmark-results.json','benchmark-selection.json']:
  p=old/name
  if p.exists():removed.append({'path':str(p),'files':1,'bytes':p.stat().st_size});bytes_total+=p.stat().st_size;p.unlink()
 for name in ['pipeline.py','launch.py']:
  p=old/name
  if p.exists():p.rename(p.with_suffix('.py.disabled'))
 (old/'status.json').write_text(json.dumps({'phase':'training_results_deleted','deleted_at':time.time(),'reason':'User requested replacement by Count/Order/State epoch-1 warm-start matched to Robometer','replacement_setup':str(ROOT/task),'raw_data_and_cache_preserved':True},indent=2)+'\n')
assert hashlib.sha256(Path(policy['initial_checkpoint']).read_bytes()).hexdigest()==policy['initial_checkpoint_sha256']
for t in policy['tasks'].values():assert (Path(t['cache_dir'])/'cache-ready.json').exists()
report={'status':'completed','deleted_at':time.time(),'removed':removed,'total_bytes':bytes_total,'source_checkpoint_preserved':True,'raw_data_and_labels_and_caches_preserved':True}
(ROOT/'cleanup.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
