"""Two isolated two-GPU lanes; all five jobs independently start from official weights."""
import argparse,copy,fcntl,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from common import *

def gpu_state():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 busy=set(x.strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines())
 return {int(c[0]):dict(occupied=c[1].strip() in busy,memory=int(c[2]),utilization=int(c[3])) for line in raw.splitlines() if (c:=line.split(','))}
def idle(gpus):
 s=gpu_state();return all(not s[g]['occupied'] and s[g]['memory']<1024 and s[g]['utilization']<10 for g in gpus)
def wait_idle(task,gpus):
 while True:
  if idle(gpus):
   time.sleep(3)
   if idle(gpus):return
  status(task,'waiting_for_gpus',gpus=gpus);time.sleep(10)
def command(script,args,ddp=False):return [PY,'-u']+(['-m','torch.distributed.run','--standalone','--nproc_per_node=2'] if ddp else [])+[str(ROOT/script)]+args
def run(task,gpus,script,args,phase,logname,ddp=False,cpu=False):
 s=ROOT/task;env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if cpu else ','.join(map(str,gpus)),PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
  OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 log=s/logname;cmd=command(script,args,ddp);status(task,phase,command=cmd,log=str(log),gpus=gpus,child_pid=None)
 with log.open('a') as f:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  status(task,phase,child_pid=child.pid)
  while child.poll() is None:
   extra={};out=Path(config(task)['output_dir'])
   if phase=='training':
    p=out/'metrics.jsonl'
    if p.exists():
     rr=rows(p)
     if rr:extra['last_training_record']=rr[-1]
    extra['saved_epochs']=[i for i in range(1,4) if (out/f"checkpoint-{config(task)['steps_per_epoch']*i}"/'complete.json').exists()]
   if phase=='evaluating':extra['evaluation']={p.stem:read(p) for p in (out/'evaluation').glob('worker-*.json')}
   status(task,phase,**extra);time.sleep(10)
 status(task,phase,child_pid=None,returncode=child.returncode)
 return child.returncode
def execute(task,gpus,recover=False):
 lock=(ROOT/task/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 c=config(task);s=ROOT/task;out=Path(c['output_dir']);policy=read(ROOT/'launch-policy.json')
 try:
  assert read(ROOT/'preflight.json')['status']=='passed'
  for n,h in read(ROOT/'preflight.json')['code_sha256'].items():assert sha(ROOT/n)==h,n
  if recover:
   actual=read(s/'execution.config.json')
  else:
   assert not out.exists(),'Do not overwrite a prior output'
   candidate=None
   for index,choice in enumerate(policy['smoke_candidates']):
    smoke=s/f'smoke-{index}';assert not smoke.exists();smoke.mkdir()
    actual=copy.deepcopy(c);actual['training'].update(per_device_train_batch_size=choice['microbatch'],gradient_accumulation_steps=choice['accumulation'],gradient_checkpointing=choice['gradient_checkpointing'])
    assert choice['microbatch']*choice['accumulation']*2==18
    test=copy.deepcopy(actual);test.update(output_dir=str(smoke/'output'),audit_dir=str(smoke));save(smoke/'config.json',test)
    wait_idle(task,gpus);rc=run(task,gpus,'train_run.py',['--config',str(smoke/'config.json'),'--smoke'],'gpu_smoke',f'smoke-{index}.log',True)
    if rc:
     text=(s/f'smoke-{index}.log').read_text()
     if 'out of memory' not in text.lower():raise RuntimeError('Smoke failed: '+str(s/f'smoke-{index}.log'))
     continue
    receipts=[read(smoke/f'smoke-rank{rank}.json') for rank in [0,1]]
    assert all(x['status']=='passed' and x['steps']==4 for x in receipts)
    if max(x['peak_reserved_gib'] for x in receipts)>77:continue
    init=[read(smoke/f'initialization-rank{rank}.json') for rank in [0,1]];assert init[0]['trainable_sha256']==init[1]['trainable_sha256']
    assert all(x['initial_checkpoint'] is None and x['fresh_lora_B_all_zero'] for x in init)
    candidate=dict(choice=choice,receipts=receipts,weights_reused=False);break
   assert candidate is not None,'No memory-safe configuration for global batch18'
   actual.update(output_dir=str(out),audit_dir=str(s));save(s/'execution.config.json',actual);save(s/'smoke-verification.json',dict(status='passed',**candidate))
   for d in s.glob('smoke-*/output'):
    if d.is_dir():shutil.rmtree(d)
  complete=out/'training-complete.json';args=['--config',str(s/'execution.config.json')]
  if not complete.exists():
   if recover:
    checkpoints=[p for p in out.glob('checkpoint-*') if (p/'complete.json').exists()]
    assert checkpoints,'No valid checkpoint for safe recovery; request action'
    last=max(checkpoints,key=lambda p:read(p/'complete.json')['step']);assert read(last/'run_config.json')==actual
    args+=['--resume',str(last)]
   wait_idle(task,gpus);status(task,'training',training_started_at=time.time(),selected_training=actual['training'])
   assert run(task,gpus,'train_run.py',args,'training','train.log',True)==0,'Training failed; checkpoint-preserving recovery required'
  status(task,'verifying',training_finished_at=time.time())
  assert run(task,gpus,'evaluate_final.py',['--config',str(s/'execution.config.json'),'--verify'],'verifying','verification.log',cpu=True)==0
  wait_idle(task,gpus)
  assert run(task,gpus,'evaluate_final.py',['--config',str(s/'execution.config.json')],'evaluating','evaluation.log',True)==0
  assert run(task,gpus,'evaluate_final.py',['--config',str(s/'execution.config.json'),'--aggregate'],'aggregating','aggregate.log',cpu=True)==0
  assert run(task,gpus,'plot_final.py',['--config',str(s/'execution.config.json')],'plotting','plot.log',cpu=True)==0
  m=read(out/'evaluation/metrics.json');manifest=read(out/'evaluation/curves/manifest.json');assert manifest['status']=='passed' and m['coverage_complete']
  status(task,'completed',finished_at=time.time(),checkpoint=m['checkpoint'],metrics=m['overall'],curve_path=str(out/'evaluation/curves/overview.png'))
 except BaseException as e:
  previous=read(s/'status.json');status(task,'failed',failed_phase=previous.get('phase'),error=repr(e),traceback=traceback.format_exc(),failed_at=time.time());return False
 return True
def claim(lane):
 with (ROOT/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for task in TASKS:
   if read(ROOT/task/'status.json')['phase']=='pending_cache':
    status(task,'claimed',lane=lane,lane_pid=os.getpid(),gpus=[lane*2,lane*2+1]);return task
 return None
def lane(lane_id):
 lock=(ROOT/f'lane-{lane_id}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while task:=claim(lane_id):execute(task,[lane_id*2,lane_id*2+1])
def main():
 lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(ROOT/'cache-status.json')['phase']=='completed' and read(ROOT/'preflight.json')['status']=='passed'
 children=[];logs=[];start=time.time()
 for i in [0,1]:
  f=(ROOT/f'lane-{i}.log').open('a');logs.append(f);children.append(subprocess.Popen([PY,'-u',str(__file__),'--lane',str(i)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
 while True:
  states={task:read(ROOT/task/'status.json') for task in TASKS};done=all(p.poll() is not None for p in children)
  failed=any(s['phase']=='failed' for s in states.values())
  phase=('completed' if all(s['phase']=='completed' for s in states.values()) else 'failed') if done else ('running_with_failure' if failed else 'running')
  save(ROOT/'status.json',dict(phase=phase,pid=os.getpid(),started_at=start,updated_at=time.time(),tasks=states,lane_pids=[p.pid for p in children]))
  if done:
   save(ROOT/'comparison.json',{task:s.get('metrics') for task,s in states.items()});return
  time.sleep(10)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lane',type=int);p.add_argument('--recover-task',choices=TASKS);p.add_argument('--gpus',type=int,nargs=2);a=p.parse_args()
 if a.lane is not None:lane(a.lane)
 elif a.recover_task:
  assert a.gpus and all(g in range(4) for g in a.gpus);execute(a.recover_task,a.gpus,recover=True)
 else:main()
