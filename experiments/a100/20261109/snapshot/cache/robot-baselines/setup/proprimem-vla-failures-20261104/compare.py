import json,subprocess
import numpy as np
from pathlib import Path
from publish import progress as label_progress
def compare(task,a,b):
 sa=[json.loads(l) for l in (Path(a['evidence'])/'states.jsonl').read_text().splitlines()]
 sb=[json.loads(l) for l in (Path(b['evidence'])/'states.jsonl').read_text().splitlines()]
 def route(states):
  q=np.asarray([s['joint_state'] for s in states],dtype=float);grid=np.linspace(0,len(q)-1,64);return np.stack([np.interp(grid,np.arange(len(q)),q[:,i]) for i in range(q.shape[1])],axis=1)
 route_mae=float(np.mean(np.abs(route(sa)-route(sb))))
 _,ea=label_progress(task,sa);_,eb=label_progress(task,sb)
 event_match=[round(e['progress'],6) for e in ea]==[round(e['progress'],6) for e in eb]
 match=route_mae<=.15 and event_match and a['seed']==b['seed'] and a['success']==b['success'] and abs(a['steps']-b['steps'])<=max(10,.2*a['steps'])
 subprocess.run(['/cache/robot-baselines/envs/simplemem-libero/bin/python','/cache/robot-baselines/setup/proprimem-vla-failures-20261104/export_validation.py'],check=True)
 return match,route_mae,event_match
