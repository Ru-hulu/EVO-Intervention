"""CPU-only cache build, two-GPU warm-start training, checkpoint audit and final curves."""
from pathlib import Path
import os,json,time,subprocess,fcntl,hashlib,traceback
S=Path(__file__).resolve().parent
PY='/cache/cvpzhr/envs/robometer/bin/python'
THIRD='/cache/cvpzhr/code/robometer-current3-w8-e1-20261104/third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def gpu_info():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 occupied=set(x.strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines())
 return {int(x[0]):{'memory_mib':int(x[2]),'utilization':int(x[3]),'occupied':x[1].strip() in occupied} for line in raw.splitlines() if (x:=line.split(','))}
def available(info):return sorted(g for g,x in info.items() if not x['occupied'] and x['memory_mib']<1024 and x['utilization']<10)
def latest_log(path):
 if not path.exists():return None
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-20000));lines=f.read().decode(errors='replace').splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except json.JSONDecodeError:pass

def main():
 lock=(S/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg=read(S/'training.config.json');out=Path(cfg['output_dir']);data=Path(cfg['train_data']).parent
 state={'pid':os.getpid(),'started_at':time.time(),'task':cfg['task'],'additional_epochs':3,'initial_checkpoint':cfg['initial_checkpoint'],'output_dir':str(out),'cache_dir':str(data),'augmentation':'none','world_size':2,'batch_per_gpu':9,'global_batch_size':18,'expected_steps':cfg['expected_steps']}
 def update(phase,**kw):state.update(phase=phase,updated_at=time.time(),**kw);save(S/'pipeline-status.json',state)
 base_env={**os.environ,'PYTHONPATH':str(S/'code')+':'+THIRD,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONNOUSERSITE':'1','NCCL_DEBUG':'WARN'}
 def run(command,log,phase,gpus=None):
  env={**base_env,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)) if gpus is not None else ''}
  update(phase,command=command,log=str(log),child_pid=None)
  with Path(log).open('a') as f:
   child=subprocess.Popen(command,cwd=S,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   update(phase,child_pid=child.pid)
   while child.poll() is None:
    extra={}
    if phase=='building_caches' and (S/'cache-controller.json').exists():extra['cache_status']=read(S/'cache-controller.json')
    if phase=='training_and_final_evaluation':
     row=latest_log(out/'metrics.jsonl')
     if row:extra['last_training_record']=row
     extra['saved_checkpoints']=[p.name for p in sorted(out.glob('checkpoint-*')) if (p/'complete.json').exists()]
    update(phase,**extra);time.sleep(10)
  update(phase,child_pid=None,returncode=child.returncode)
  if child.returncode:raise RuntimeError(f'{phase} exited {child.returncode}; see {log}')
 try:
  update('preflight');assert read(S/'source-readiness.json')['status']=='passed'
  for p,digest in read(S/'run-input-sha256.json').items():assert sha(p)==digest,p
  assert not out.exists() or not any(out.iterdir()),'Refusing to overwrite an existing training run'
  run([PY,'-u',str(S/'run_cache.py')],S/'cache-controller.log','building_caches')
  ready=read(data/'cache-ready.json');assert ready['status']=='ready' and ready['augmentation']=='none' and ready['train_samples']==7769 and ready['validation_samples']==1744
  run([PY,str(S/'preflight_training.py')],S/'training-preflight.log','validating_training_inputs')
  update('waiting_for_two_idle_gpus')
  while True:
   info=gpu_info();free=available(info)
   if len(free)>=2:
    chosen=free[:2];time.sleep(5)
    if all(g in available(gpu_info()) for g in chosen):break
   update('waiting_for_two_idle_gpus',gpu_state=info);time.sleep(10)
  state.update(gpu_ids=chosen,training_started_at=time.time())
  command=[PY,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(S/'train_from_checkpoint.py'),'--config',str(S/'training.config.json')]
  run(command,S/'train.log','training_and_final_evaluation',chosen)
  report=read(out/'report.json');assert report['status']=='completed' and report['steps']==cfg['expected_steps'] and report['validation_samples']==1744
  state['training_and_evaluation_finished_at']=time.time()
  run([PY,str(S/'verify_final.py')],S/'verification.log','verifying_results')
  # Use the already verified final predictions to draw every validation curve, without another GPU pass.
  run([PY,'-c',"import json; from pathlib import Path; import postprocess_validation as p; c=p.read_json(p.SETUP/'training.config.json'); r=p.render(c); p.write_json(p.SETUP/'curves-ready.json',r); print(json.dumps(r))"],S/'plotting.log','plotting_validation')
  curves=read(S/'curves-ready.json');assert curves['episodes']==7 and curves['samples']==1744
  update('completed',finished_at=time.time(),final_checkpoint=report['final_checkpoint'],metrics=read(out/'evaluation/verified-metrics.json'),curve_path=curves['overview'],curve_bundle=curves['archive'],saved_epochs=[1,2,3])
 except BaseException as e:update('failed',failed_phase=state.get('phase'),error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
