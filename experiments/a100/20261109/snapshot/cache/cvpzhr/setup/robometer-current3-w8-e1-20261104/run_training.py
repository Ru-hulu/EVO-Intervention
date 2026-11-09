"""User-authorized one-epoch Robometer training, with pre-validation recovery checkpoints."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import fcntl,hashlib,json,os,signal,subprocess,time,traceback,shutil
SETUP=Path(__file__).resolve().parent
CODE=Path('/cache/cvpzhr/code/robometer-current3-w8-e1-20261104')
PY='/cache/cvpzhr/envs/robometer/bin/python'
CONFIG=SETUP/'config.json'
C=json.loads(CONFIG.read_text())
OUT=Path(C['output_dir'])
ENV={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'),
     'CUDA_VISIBLE_DEVICES':'0,1,2,3','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
     'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
     'PYTHONNOUSERSITE':'1','NCCL_DEBUG':'WARN'}
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def atomic_json(path,value):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,indent=2));temp.replace(path)
lock=(SETUP/'training.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(pid=os.getpid(),started_at=now(),started_timestamp=time.time(),
           output=str(OUT),code=str(CODE),config=str(CONFIG),epochs=1,expected_steps=9967,
           validation_steps=[4984,9967],formal_training_started=False,
           save_before_validation=True,world_size=4,global_batch=32)
child=None
def save(phase,**kw):
 state.update(phase=phase,updated_at=now(),**kw);atomic_json(SETUP/'training-status.json',state)
def stop_child():
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def interrupted(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
def records(path):
 if not path.exists():return []
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
try:
 save('prelaunch')
 assert json.loads((SETUP/'preflight-status.json').read_text())['phase']=='completed'
 assert not OUT.exists() or not any(OUT.iterdir()),'Formal output is nonempty; refusing to overwrite'
 assert C['training']['num_train_epochs']==1 and C['training']['learning_rate']==1e-4
 assert C['eval_steps']==4984 and C['training'].get('max_steps',-1) in [-1,None]
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest
            for p,digest in json.loads((SETUP/'code-sha256.json').read_text()).items())
 data=Path(C['train_data']).parent
 report=json.loads((data/'report.json').read_text())
 assert report['status']=='completed'
 for split,path in [('train',C['train_data']),('validation',C['validation_data'])]:
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==report['splits'][split]['sha256']
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 assert not apps,'GPUs occupied before formal launch'
 atomic_json(SETUP/'training-inputs.json',dict(config=C,config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
     source='official Robometer-4B; fresh LoRA; official progress head retained',
     smoke_checkpoint_used=False,train_samples=318931,validation_samples=66891,
     authorization='User explicitly requested starting training'))
 with (SETUP/'training.log').open('a') as log:
  child=subprocess.Popen([PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
       '-m','robometer_training.train','--config',str(CONFIG)],cwd=CODE,env=ENV,
       stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  save('initializing',child_pid=child.pid,formal_training_started=True,launched_at=now())
  while child.poll() is None:
   logs=records(OUT/'metrics.jsonl');events=records(OUT/'lifecycle.jsonl')
   latest=logs[-1] if logs else {}
   if events and events[-1]['event']=='validation_started':phase='validating'
   elif latest:phase='training'
   else:phase='initializing'
   save(phase,step=latest.get('step',0),epoch=latest.get('epoch',0),
        latest_metrics=latest,last_lifecycle=events[-1] if events else None)
   if shutil.disk_usage(OUT.parent).free<5*1024**3:raise RuntimeError('Disk free below 5 GiB')
   time.sleep(15)
 if child.returncode:raise RuntimeError(f'training exited {child.returncode}; see training.log')
 trainer=json.loads((OUT/'trainer_state.json').read_text())
 assert trainer['global_step']==9967 and abs(trainer['epoch']-1)<1e-6
 validations=[]
 for step in [4984,9967]:
  checkpoint=OUT/f'checkpoint-{step}'
  receipt=json.loads((checkpoint/'complete.json').read_text())
  assert receipt['complete'] and receipt['saved_before_validation']
  metrics=json.loads((checkpoint/'validation.json').read_text())
  assert metrics['eval_samples']==66891
  assert sum(metrics['eval_'+suite+'_samples'] for suite in ['Count','Order','State'])==66891
  validations.append(dict(step=step,epoch=receipt['epoch'],checkpoint=str(checkpoint),metrics=metrics))
 best=json.loads((OUT/'best.json').read_text())
 last=json.loads((OUT/'last.json').read_text())
 summary=dict(status='completed',started_at=state['started_at'],completed_at=now(),
      elapsed_seconds=time.time()-state['started_timestamp'],epochs=1,steps=9967,
      train_samples=318931,validation_samples=66891,validations=validations,best=best,last=last,
      initialization='official Robometer-4B and fresh LoRA',config=C)
 atomic_json(OUT/'report.json',summary);atomic_json(SETUP/'training-completed.json',summary)
 save('completed',child_pid=None,step=9967,epoch=1,report=str(OUT/'report.json'))
except BaseException as exc:
 stop_child()
 last=json.loads((OUT/'last.json').read_text()) if (OUT/'last.json').exists() else None
 save('failed',error=repr(exc),traceback=traceback.format_exc(),last_recovery_checkpoint=last)
 raise
