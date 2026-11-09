"""Resident-model paired render validation followed by per-task failure quotas."""
import os,sys,json,time,runpy,subprocess,socket,traceback,signal
import numpy as np
from pathlib import Path
S=Path(__file__).parent;R=Path('/cache/robot-baselines');MODEL=os.environ['VLA_WORKER_MODEL'];GPU=os.environ['VLA_WORKER_GPU']
os.environ.update(CUDA_VISIBLE_DEVICES=GPU,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HOME=str(R/'.cache/huggingface'),HF_HUB_CACHE=str(R/'.cache/huggingface/hub'),OPENPI_DATA_HOME=str(R/'.cache/openpi'))
sys.path.insert(0,str(S));import capture
from publish import publish,TASKS
ORDER=['cover_blocks','blocks_ranking_try'];TASKS=sorted([t for t in TASKS if t['sim_task'] in ORDER],key=lambda t:ORDER.index(t['sim_task']))
ROOT=R/'eval-results/imprint-extra-20261105';VALID=R/'eval-results/vla-render-validation-20261104';SERVER=None

def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def simple_init():
 os.chdir(R/'repos/SimpleMemVLA');sys.path.insert(0,os.getcwd())
 import rmbench_sim.eval_success as ev
 os.chdir(R/'repos/SimpleMemVLA/rmbench_sim/rmbench');capture.install_simplemem();os.chdir(R/'repos/SimpleMemVLA');cached=[];build=ev.build_policy
 def resident(args):
  if not cached:cached.append(build(args))
  return cached[0]
 ev.build_policy=resident
 return ev
EV=None
class Persistent:
 def __init__(self,p):self.p=p
 def poll(self):return self.p.poll()
 def terminate(self):pass
 def kill(self):pass
 def wait(self,timeout=None):return 0
ORIGINAL_POPEN=subprocess.Popen
def popen(*args,**kwargs):
 global SERVER
 cmd=args[0] if args else kwargs.get('args',[])
 if any('serve_robomme_tcp.py' in str(a) for a in cmd):
  if SERVER is None:SERVER=ORIGINAL_POPEN(*args,**kwargs)
  if SERVER.poll() is not None:raise RuntimeError('resident policy server exited')
  return Persistent(SERVER)
 return ORIGINAL_POPEN(*args,**kwargs)

def episode(task,seed,spp,out):
 global EV
 meta=out/'result.json'
 if meta.exists():return json.loads(meta.read_text())
 out.mkdir(parents=True,exist_ok=True);evidence=out/'evidence';os.environ.update(VLA_RENDER_SPP=str(spp),VLA_CAPTURE_DIR=str(evidence))
 started=time.time()
 if MODEL=='simplemem':
  if EV is None:EV=simple_init()
  sys.argv=['eval','--pretrained_checkpoint',str(R/'models/SimpleMemVLA/RMBench'),'--tasks',task,'--group_size','1','--seeds_per_task','1','--base_seed',str(seed),'--no_seed_cache','--instruction_type','unseen']
  args=EV.parse_args();pools=[];original=EV.InProcSimPool
  class Pool(original):
   def __init__(self,*a,**kw):super().__init__(*a,**kw);pools.append(self)
  EV.InProcSimPool=Pool
  try:
   result=EV.evaluate_tasks(args,[task],{});assert result['n'][task]==1,result
  finally:
   capture.close()
   for pool in pools:
    robots=[getattr(service.env.TASK_ENV,'robot',None) for service in pool.services if service.env is not None]
    pool.close()
    for robot in robots:
     for side in ('left','right'):
      proc=getattr(robot,side+'_proc',None)
      if proc is not None and proc.is_alive():proc.terminate();proc.join(timeout=5)
   EV.InProcSimPool=original
 else:
  os.chdir(R/'repos/imprint');subprocess.Popen=popen
  sys.argv=['run_local','--rmbench-root',str(R/'dependencies/EventVLA/RoboTwin-Mem'),'--checkpoint',str(R/f'models/IMPRINT/simulation/rmbench/{MODEL}/stage2/step-100000'),'--dinov3-model',str(R/'models/DINOv3-ViT-B16'),'--architecture',MODEL,'--task',task,'--trials','1','--start-seed',str(seed),'--output',str(out/'run'),'--policy-python',str(R/'envs/imprint/bin/python'),'--sim-python',str(S/'sim-python'),'--policy-gpu',GPU,'--sim-gpu',GPU,'--port',str(8070+int(GPU))]
  runpy.run_path(str(R/'repos/imprint/benchmarks/rmbench/eval/run_local.py'),run_name='__main__');evidence=out/'run/evidence/0'
 e=json.loads((evidence/'evidence.json').read_text());assert e['frames']>0
 import av
 for camera in capture.CAMERAS:
  with av.open(str(evidence/(camera+'.mp4'))) as video:
   stream=video.streams.video[0];frames=stream.frames or sum(1 for _ in video.decode(video=0));assert frames==e['frames'],(frames,e['frames'])
 x=dict(status='complete',model=MODEL,task=task,seed=e['seed'],requested_seed=seed,success=e['success'],steps=e['steps'],seconds=time.time()-started,evidence=str(evidence),spp=spp);save(meta,x);return x

def main():
 while not (S/'collection_config.json').exists():time.sleep(10)
 config=json.loads((S/'collection_config.json').read_text());mode=config['mode']
 if config.get('renderer_spp') is not None:
  spp=int(config['renderer_spp']);assert spp in (8,32)
  save(S/f'{MODEL}-render-setting.json',dict(renderer_spp=spp,source='explicit_user_configuration',updated=time.time(),validation_history_preserved=True))
 else:
  checks=[]
  for t in TASKS:
   a=episode(t['sim_task'],100000,32,VALID/MODEL/t['sim_task']/'rt32');b=episode(t['sim_task'],100000,8,VALID/MODEL/t['sim_task']/'rt8')
   import importlib,compare as comparison
   match,route_mae,event_match=importlib.reload(comparison).compare(t['sim_task'],a,b)
   checks.append(dict(task=t['sim_task'],passed=match,joint_route_mae=route_mae,progress_event_sequence_match=event_match,baseline=a,candidate=b));save(VALID/MODEL/'validation.json',dict(status='running',checks=checks))
  passed=all(x['passed'] for x in checks);spp=8 if passed else 32
  save(VALID/MODEL/'validation.json',dict(status='passed' if passed else 'rejected_using_32',selected_spp=spp,checks=checks,criterion='same accepted scene seed, native outcome and progress event sequence; 64-point resampled joint-route MAE <=0.15; action count within 20% or 10 actions; seven paired task episodes. Pilot check, not proof of identical policies.'))
 while True:
  progress=False
  for t in TASKS:
   task=t['sim_task'];folder=ROOT/MODEL/task;records=[json.loads(p.read_text()) for p in folder.glob('seed_*/status.json')];records=[x for x in records if x.get('execution_host','Huawei')=='Huawei'];failures=sum(x['status']=='complete' and not x['success'] for x in records)
   completed=sum(x['status']=='complete' for x in records)
   if (failures if mode=='failures' else completed)>=(10 if MODEL=='full_attention' else 20):continue
   progress=True;actual=[x.get('seed',x['requested_seed']) for x in records];seed=max([101999]+actual)+1;out=folder/f'seed_{seed}';save(out/'status.json',dict(status='running',task=task,model=MODEL,requested_seed=seed,spp=spp,start=time.time()))
   try:
    x=episode(task,seed,spp,out);save(out/'status.json',x)
    import importlib,publish as publisher
    importlib.reload(publisher).publish()
   except Exception as error:
    if (out/'result.json').exists():
     save(out/'status.json',json.loads((out/'result.json').read_text()))
     save(out/'export_error.json',dict(error=repr(error)))
    else:save(out/'status.json',dict(status='error',task=task,model=MODEL,requested_seed=seed,seed=seed,error=repr(error)))
    raise
  if not progress:break
 save(S/f'{MODEL}-done.json',dict(status='complete',target_episodes=(20 if MODEL=='full_attention' else 40)))
if __name__=='__main__':
 try:main()
 except BaseException:
  save(S/f'{MODEL}-error.json',dict(status='error',traceback=traceback.format_exc()));raise
 finally:
  if SERVER is not None:SERVER.terminate();SERVER.wait(timeout=30)
