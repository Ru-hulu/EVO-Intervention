import subprocess,os,time,json,fcntl
from pathlib import Path
p=Path('/cache/robot-baselines/setup/robomme-coin-20261105')
lock=(p/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (p/'preflight.json').exists()
gpu='1';env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
(p/'selected-gpu.json').write_text(json.dumps({'gpu':1,'mode':'shared GPU explicitly authorized','jax_memory_fraction':0.60,'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'pid':os.getpid()},indent=2))
(p/'status.json').write_text(json.dumps({'stage':'extracting_history_features','gpu':1,'shared':True,'pid':os.getpid()},indent=2))
with (p/'pipeline.log').open('a') as log:
 result=subprocess.run(['bash',str(p/'launch.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
 old=json.loads((p/'status.json').read_text());old.update(stage='failed',exit_code=result.returncode)
 (p/'status.json').write_text(json.dumps(old,indent=2))
raise SystemExit(result.returncode)
