"""Recover RoboMME subtask boundaries by replaying saved actions; verify qpos."""
import json,sys,time,traceback
from pathlib import Path
import numpy as np
from env_runner import EnvRunner
ROOT=Path('/cache/robot-baselines'); OUT=ROOT/'annotations/boundary-integral-20261105'
PAIRS=[('RoboMME-FrameSamp-20261105','robomme-closed-loop-20261105'),('RoboMME-SimpleMem-20261105','simplemem-robomme-closed-loop-20261105')]
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False));t.replace(p)
def main():
 for ds,run in PAIRS:
  for row in map(json.loads,(ROOT/'viewer-datasets'/ds/'metadata.jsonl').read_text().splitlines()):
   dest=OUT/ds/'replay'/(row['sample_id']+'.json')
   if dest.exists():continue
   source=ROOT/'eval-results'/run/row['task_id']/f"ep{row['episode_index']:03}"
   result=json.loads((source/'result.json').read_text());runner=None;started=time.time()
   try:
    runner=EnvRunner(row['task_id'],OUT/'replay-scratch',max_steps=1300);runner.make_env(row['episode_index']);initial=runner.get_init_obs();env=runner.env.unwrapped
    tasklist=[{'index':i,'name':t['name'],'demonstration':bool(t.get('demonstration',False))} for i,t in enumerate(env.task_list)]
    def state(frame):return dict(frame=frame,index=int(env.timestep),name=str(getattr(env,'current_task_name_online','')),failure=bool(getattr(env,'current_task_failure',False)))
    events=[state(len(initial['images'])-1)];maximum=0.
    for line in (source/'actions.jsonl').read_text().splitlines():
     a=json.loads(line);obs,stop,status=runner.step(np.asarray(a['action']));gap=float(np.max(np.abs(obs[2]-a['state'])));maximum=max(maximum,gap)
     if gap>1e-4:raise ValueError(f"replay mismatch frame {a['frame']}: qpos max diff {gap}")
     event=state(a['frame'])
     if any(event[k]!=events[-1][k] for k in ['index','failure']):events.append(event)
     if status!=a['status']:raise ValueError(f"status mismatch frame {a['frame']}: {status} vs {a['status']}")
    write(dest,dict(sample_id=row['sample_id'],task_list=tasklist,events=events,final_status=status,max_qpos_error=maximum,frames=row['total_frames'],execution_start=len(initial['images'])-1,seconds=time.time()-started))
    print('DONE',ds,row['sample_id'],len(events),maximum,round(time.time()-started,2),flush=True)
   except Exception:
    write(dest.with_suffix('.error.json'),dict(error=traceback.format_exc()));raise
   finally:
    if runner:runner.close_env()
   if '--one' in sys.argv:return
if __name__=='__main__':main()
