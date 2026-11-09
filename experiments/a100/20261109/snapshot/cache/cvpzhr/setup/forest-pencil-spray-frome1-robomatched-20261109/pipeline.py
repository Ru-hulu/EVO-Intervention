from pathlib import Path
import os,sys,json,time,subprocess,fcntl,traceback,hashlib,math,shutil
ROOT=Path(__file__).resolve().parent
PY='/cache/cvpzhr/envs/forest/bin/python'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def records(p):
 if not p.exists():return []
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def gpu_state():
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 occupied=set(x.strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines())
 return {int(c[0]):{'memory_mib':int(c[2]),'utilization':int(c[3]),'occupied':c[1].strip() in occupied} for l in raw.splitlines() if (c:=l.split(','))}
def lane(task):
 policy=read(ROOT/'launch-policy.json');t=policy['tasks'][task];S=Path(t['setup_dir']);O=Path(t['output_dir']);E=Path(t['evaluation_dir']);gpus=t['gpu_ids']
 lock=(S/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state={'task':task,'pid':os.getpid(),'started_at':time.time(),'initial_checkpoint':policy['initial_checkpoint'],'output_dir':str(O),'evaluation_dir':str(E),'gpu_ids':gpus,'additional_epochs':3}
 def update(phase,**kw):state.update(phase=phase,updated_at=time.time(),**kw);save(S/'status.json',state)
 env=dict(os.environ,PYTHONPATH=str(S/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)))
 def run(cmd,log,phase,use_gpu=True):
  update(phase,command=cmd,log=str(log),child_pid=None)
  with log.open('a') as f:
   child=subprocess.Popen(cmd,cwd=S,env=env if use_gpu else dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   update(phase,child_pid=child.pid)
   while child.poll() is None:
    extra={}
    if phase=='training':
     rr=records(O/'train.jsonl')
     if rr:extra['last_training_record']=rr[-1]
     extra['saved_epochs']=[i for i in range(1,4) if (O/f'epoch-{i:02d}.pt').exists()]
    elif phase=='evaluating_epoch3':extra['evaluated_samples']=sum(read(p).get('samples',0) for p in E.glob('worker-*.json'))
    update(phase,**extra);time.sleep(10)
  update(phase,child_pid=None,returncode=child.returncode)
  if child.returncode:raise RuntimeError(f'{phase} exited {child.returncode}; {log}')
 def tr(target,args,module=False):return [PY,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2']+(['-m'] if module else [])+[target]+args
 def wait_idle(phase):
  while True:
   info=gpu_state()
   if all(not info[g]['occupied'] and info[g]['memory_mib']<1024 and info[g]['utilization']<10 for g in gpus):
    time.sleep(5);again=gpu_state()
    if all(not again[g]['occupied'] and again[g]['memory_mib']<1024 and again[g]['utilization']<10 for g in gpus):return
   update(phase,gpu_state=info);time.sleep(10)
 try:
  update('preflight');assert read(ROOT/'preflight.json')['status']=='passed';assert read(ROOT/'cleanup.json')['status']=='completed'
  for rel,digest in read(ROOT/'run-files-sha256.json').items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
  assert not O.exists(),'Existing output requires an explicit recovery audit'
  cfg=read(S/'training.config.json');wait_idle('waiting_for_two_gpus')
  smoke=S/'smoke';smoke.mkdir();smoke_cfg=dict(cfg,max_steps=4,log_every_steps=1,output_dir=str(smoke/'output'),initialization_audit_dir=str(smoke))
  save(smoke/'config.json',smoke_cfg)
  run(tr('mem_lite.train_stage3',['--config',str(smoke/'config.json')],True),smoke/'run.log','gpu_smoke')
  rr=records(smoke/'output/train.jsonl');assert len(rr)==4 and rr[-1]['step']==4 and all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rr)
  assert all(r['samples']==cfg['batch_size']*cfg['gradient_accumulation_steps']*2 for r in rr)
  assert all(read(smoke/f'initialization-rank{r}.json')['all_trainable_tensors_exactly_restored'] for r in range(2))
  peak=max(max(r['peak_reserved_gib_per_gpu']) for r in rr);assert peak<74
  save(S/'smoke-verification.json',{'status':'passed','steps':4,'peak_reserved_gib':peak,'global_batch_size':rr[-1]['samples'],'weights_reused':False,'last_step_seconds':rr[-1]['step_seconds']})
  shutil.rmtree(smoke/'output')
  wait_idle('waiting_for_training_gpus');state['training_started_at']=time.time()
  run(tr('mem_lite.train_stage3',['--config',str(S/'training.config.json')],True),S/'train.log','training')
  state['training_finished_at']=time.time()
  for rank in range(2):assert read(S/f'initialization-rank{rank}.json')['all_trainable_tensors_exactly_restored']
  run([PY,str(S/'verify_checkpoints.py')],S/'checkpoint-verification.log','verifying_checkpoints',False)
  wait_idle('waiting_for_evaluation_gpus');state['evaluation_started_at']=time.time()
  args=['--checkpoint',str(O/'epoch-03.pt'),'--output',str(E),'--batch-size','4','--workers','4']
  run(tr(str(S/'evaluate_final.py'),args),S/'evaluation.log','evaluating_epoch3')
  run([PY,str(S/'evaluate_final.py'),*args,'--aggregate'],S/'evaluation-aggregate.log','aggregating',False)
  m=read(E/'metrics.json');assert m['coverage_complete'] and m['overall']['episodes']==t['counts'][2] and m['overall']['frames']==t['counts'][3]
  run([PY,str(S/'plot_validation.py')],S/'plotting.log','plotting',False)
  assert read(E/'curves/manifest.json')['status']=='passed'
  update('completed',finished_at=time.time(),metrics=m['overall'],checkpoint=str(O/'epoch-03.pt'),curve_path=str(E/'curves/overview.png'),saved_epochs=[1,2,3])
 except BaseException as exc:
  update('failed',failed_phase=state.get('phase'),error=repr(exc),traceback=traceback.format_exc());raise
def main():
 lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 children={};opened=[];start=time.time()
 for task in read(ROOT/'launch-policy.json')['tasks']:
  log=(ROOT/task/'pipeline.log').open('a');opened.append(log)
  children[task]=subprocess.Popen([sys.executable,'-u',str(__file__),'--task',task],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while True:
  states={task:read(ROOT/task/'status.json') if (ROOT/task/'status.json').exists() else {'phase':'starting'} for task in children}
  done=all(c.poll() is not None for c in children.values());failed=any(s['phase']=='failed' for s in states.values())
  save(ROOT/'status.json',{'pid':os.getpid(),'started_at':start,'updated_at':time.time(),'phase':('failed' if failed else 'completed') if done else ('running_with_failure' if failed else 'running'),'tasks':states})
  if done:return
  time.sleep(10)
if __name__=='__main__':
 if '--task' in sys.argv:lane(sys.argv[sys.argv.index('--task')+1])
 else:main()
