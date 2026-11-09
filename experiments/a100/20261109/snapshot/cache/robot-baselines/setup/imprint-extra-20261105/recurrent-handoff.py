"""Pause at an episode boundary and release the Huawei recurrent worker."""
from pathlib import Path
import json,time,subprocess,os,signal
base=Path('/cache/robot-baselines');setup=base/'setup/imprint-extra-20261105'
while not (setup/'recurrent-paused.json').exists():time.sleep(10)
pid=json.loads((setup/'recurrent-paused.json').read_text())['pid']
assert 'imprint-extra-20261105/worker.py' in Path(f'/proc/{pid}/cmdline').read_text()
children=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]:
 p,parent,args=line.split(None,2)
 if int(parent)==pid and 'serve_robomme_tcp.py' in args:children.append(int(p))
os.kill(pid,signal.SIGTERM)
for p in children:
 try:os.kill(p,signal.SIGTERM)
 except ProcessLookupError:pass
counts={'cover_blocks':0,'blocks_ranking_try':0}
for p in (base/'eval-results/imprint-extra-20261105/recurrent').glob('*/seed_*/result.json'):
 x=json.loads(p.read_text())
 if x['task'] in counts:counts[x['task']]+=1
allocation={host:{task:((20-n+1)//2 if host=='4090C' else (20-n)//2) for task,n in counts.items()} for host in ['4090C','4090D']}
report=dict(status='ready',completed_huawei=counts,remaining=allocation,time=time.time())
(setup/'recurrent-migration.json').write_text(json.dumps(report,indent=2))
p=setup/'allocation.json';data=json.loads(p.read_text());data['recurrent']={'Huawei':counts,**allocation};p.write_text(json.dumps(data,indent=2))
print(json.dumps(report),flush=True)
