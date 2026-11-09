import json,time,traceback,os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from env_runner import EnvRunner
from utils import TASK_NAME_LIST,pack_buffer,TASK_WITH_VIDEO_DEMO
from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy
ROOT=Path('/cache/robot-baselines'); OUT=ROOT/'eval-results/robomme-closed-loop-20261105'; DS=ROOT/'viewer-datasets/RoboMME-FrameSamp-20261105'
MODEL='MME-VLA FrameSamp+Modul';OUT.mkdir(parents=True,exist_ok=True);DS.mkdir(parents=True,exist_ok=True)
def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def publish():
 rows=[]
 for f in sorted(OUT.glob('*/*/result.json')):
  r=json.loads(f.read_text())
  if r['status'] not in ['success','fail','timeout']:continue
  key=r['task']+'-ep'+str(r['episode']).zfill(3);folder=DS/key;folder.mkdir(exist_ok=True)
  views={}
  for view in ['main_view','wrist_view']:
   src=f.parent/(view+'.mp4');dst=folder/src.name
   if not dst.exists():os.link(src,dst)
   views[view]={'video_path':str(dst.relative_to(DS))}
  rows.append(dict(sample_id=key,task_id=r['task'],task_name=r['task'],model_name=MODEL,episode_index=r['episode'],task_success=r['status']=='success',fps=30,total_frames=r['frames'],video_path=views['main_view']['video_path'],video_views=views,annotation_path=None,task_prompt=r['prompt'],failure_reason='原生判定：'+r['status'],evaluation_split='test',model_seed=7,checkpoint_step=79999,demonstration_frames=r['demonstration_frames']))
 tmp=DS/'metadata.tmp';tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(DS/'metadata.jsonl')
 write(OUT/'summary.json',{'target':160,'completed':len(rows),'successes':sum(r['task_success'] for r in rows),'by_task':{t:{'completed':sum(r['task_id']==t for r in rows),'successes':sum(r['task_id']==t and r['task_success'] for r in rows)} for t in TASK_NAME_LIST}})
client=MMEVLAWebsocketClientPolicy('127.0.0.1',8015)
for episode in range(10):
 for task in TASK_NAME_LIST:
  folder=OUT/task/f'ep{episode:03}';folder.mkdir(parents=True,exist_ok=True)
  if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text()).get('status') in ['success','fail','timeout']:continue
  write(OUT/'status.json',{'status':'running','task':task,'episode':episode,'time':time.time()})
  runner=None;writers=[];started=time.time();frames=0;steps=0
  try:
   response=client.reset()
   if not response.get('reset_finished'):raise RuntimeError('Policy reset did not finish')
   runner=EnvRunner(task,folder,max_steps=1300);runner.make_env(episode);initial=runner.get_init_obs();prompt=initial['task_goal']
   writers=[imageio.get_writer(folder/(name+'.mp4'),fps=30,codec='libx264',quality=7) for name in ['main_view','wrist_view']]
   for a,b in zip(initial['images'],initial['wrist_images']):
    writers[0].append_data(a);writers[1].append_data(b);frames+=1
   images=list(initial['images']);states=list(initial['states']);exec_start=len(images)-1
   demo_frames=exec_start if task in TASK_WITH_VIDEO_DEMO else 0
   img,wrist,state=initial['images'][-1],initial['wrist_images'][-1],initial['states'][-1]
   outcome='unknown';stop=False
   with (folder/'actions.jsonl').open('w') as actionlog:
    while not stop:
     response=client.add_buffer(pack_buffer(images,states,exec_start))
     if not response.get('add_buffer_finished'):raise RuntimeError('History buffer did not finish')
     actions=client.infer({'observation/image':img,'observation/wrist_image':wrist,'observation/state':state,'prompt':prompt})['actions'][:16]
     if not np.isfinite(actions).all():raise ValueError('Non-finite model action')
     images=[];states=[];exec_start=0
     for action in actions:
      obs,stop,outcome=runner.step(action);steps+=1
      if outcome=='error' or obs[0] is None:raise RuntimeError('Simulation step failed')
      img,wrist,state=obs
      images.append(img.copy());states.append(state.copy());writers[0].append_data(img);writers[1].append_data(wrist);frames+=1
      actionlog.write(json.dumps({'step':steps,'frame':frames-1,'state':state.tolist(),'action':action.tolist(),'status':outcome})+'\n')
      if steps>1300:stop=True;outcome='timeout'
      if stop:break
   if outcome not in ['success','fail','timeout']:raise RuntimeError('Unexpected final status: '+str(outcome))
   for w in writers:w.close()
   writers=[]
   write(folder/'result.json',{'task':task,'episode':episode,'status':outcome,'steps':steps,'frames':frames,'prompt':prompt,'demonstration_frames':demo_frames,'seconds':time.time()-started})
   publish();print(task,episode,outcome,steps,round(time.time()-started,1),flush=True)
  except Exception:
   write(folder/'error.json',{'task':task,'episode':episode,'error':traceback.format_exc(),'time':time.time()});traceback.print_exc()
   write(OUT/'status.json',{'status':'error','task':task,'episode':episode});raise
  finally:
   for w in writers:w.close()
   if runner is not None:runner.close_env()
write(OUT/'status.json',{'status':'complete','time':time.time()})
client._ws.close()
