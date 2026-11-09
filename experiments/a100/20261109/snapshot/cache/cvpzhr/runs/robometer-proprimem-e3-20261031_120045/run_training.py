"""Authorized 3-epoch run, final-only checkpoint and final full validation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import os,json,time,subprocess,signal,shutil,fcntl,math,traceback
RUN=Path(__file__).resolve().parent;INFO=json.loads((RUN/'run-info.json').read_text());CODE=Path(INFO['code']);OUT=Path(INFO['output']);PY=INFO['python']
ENV={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'),'CUDA_VISIBLE_DEVICES':'0,1,2,3','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','PYTHONNOUSERSITE':'1','NCCL_DEBUG':'WARN'}
lock=(RUN/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
state={'pid':os.getpid(),'started_at':now(),'run':str(RUN),'output':str(OUT),'epochs':3,'expected_updates':11856,'downloads_to_local':False,'shutdown_authorized':False}
def save(phase,**kw):
 state.update(phase=phase,updated_at=now(),**kw);p=RUN/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(RUN/'status.json')
def gpu():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
 values=[list(map(int,line.split(','))) for line in text.strip().splitlines()]
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
 return dict(timestamp=time.time(),phase=state.get('phase'),gpus=[dict(index=a,utilization=b,memory_mib=c,total_mib=d) for a,b,c,d in values],compute_processes=procs)
def idle():
 s=gpu();return not s['compute_processes'] and all(g['utilization']<=5 and g['memory_mib']<1024 for g in s['gpus'])
def wait_idle(next_phase):
 save('waiting_for_idle_gpu',next_phase=next_phase)
 while True:
  if idle():
   time.sleep(15)
   if idle():return
  time.sleep(15)
def stop_own_process(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def run_stage(phase,args,log):
 wait_idle(phase);started=time.time()
 with (RUN/log).open('a') as stream:
  p=subprocess.Popen([PY,*args],cwd=CODE,env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  save(phase,child_pid=p.pid,stage_started_at=now(),stage_started_epoch=started)
  try:
   while p.poll() is None:
    with (RUN/'gpu.jsonl').open('a') as f:f.write(json.dumps(gpu())+'\n')
    if shutil.disk_usage(OUT.parent).free<2*1024**3:stop_own_process(p);raise RuntimeError('Disk free below 2 GiB; only this stage stopped')
    time.sleep(15)
  except BaseException:stop_own_process(p);raise
 if p.returncode:raise RuntimeError(f'{phase} exited with status {p.returncode}; see {log}')
 state[phase+'_seconds']=time.time()-started;state[phase+'_finished_at']=now();return state[phase+'_seconds']
try:
 if (RUN/'completed.json').exists():raise RuntimeError('Run already completed; refusing to repeat')
 save('starting')
 run_stage('training',['-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','robometer_training.train','--config',str(RUN/'config.json')],'training.log')
 checkpoints=sorted(OUT.glob('checkpoint-*'));assert len(checkpoints)==1,'Expected only the final checkpoint'
 checkpoint=checkpoints[0];trainer=json.loads((OUT/'trainer_state.json').read_text())
 assert trainer['global_step']==INFO['total_steps'] and abs(trainer['epoch']-3)<1e-6,'Unexpected training extent'
 assert checkpoint.name=='checkpoint-'+str(trainer['global_step'])
 state['checkpoint']=str(checkpoint)
 validation=OUT/'validation-03'
 run_stage('validation',['-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(RUN/'validate.py'),'--config',str(RUN/'config.json'),'--adapter',str(checkpoint),'--output',str(validation)],'validation.log')
 shards=[json.loads((validation/f'metrics-rank{r}.json').read_text()) for r in range(4)];n=sum(s['samples'] for s in shards);assert n==INFO['validation_samples']
 predictions=[]
 for rank in range(4):predictions.extend(json.loads(line) for line in (validation/f'predictions-rank{rank}.jsonl').read_text().splitlines())
 predictions.sort(key=lambda r:r['sample_index']);assert [p['sample_index'] for p in predictions]==list(range(n));assert len({p['id'] for p in predictions})==n
 with (validation/'predictions.jsonl').open('w') as f:
  for record in predictions:f.write(json.dumps(record,ensure_ascii=False)+'\n')
 metrics=dict(samples=n,loss=sum(s['loss_sum'] for s in shards)/n,mae_pp=sum(s['absolute_error_sum_pp'] for s in shards)/n,rmse_pp=math.sqrt(sum(s['squared_error_sum_pp2'] for s in shards)/n),bias_pp=sum(s['bias_sum_pp'] for s in shards)/n,seconds=state['validation_seconds'],peak_allocated_gib=max(s['peak_allocated_gib'] for s in shards),conditioning='5 FPS causal main-view current and preceding 7 observations; no future frames',throughput_note='Four-GPU batch evaluation; not single-frame online latency')
 (validation/'metrics.json').write_text(json.dumps(metrics,indent=2))
 logs=[json.loads(line) for line in (OUT/'metrics.jsonl').read_text().splitlines()];losses=[x for x in logs if 'loss' in x];summaries=[x for x in logs if 'train_loss' in x]
 grads=[x['grad_norm'] for x in logs if 'grad_norm' in x];assert all(math.isfinite(x) for x in grads)
 report=dict(status='completed',model='Robometer-4B',epochs=3,updates=trainer['global_step'],batch_per_gpu=8,global_batch=32,gradient_accumulation_steps=1,checkpoint=str(checkpoint),training=dict(seconds=state['training_seconds'],last_logged_step=losses[-1]['step'],last_logged_loss=losses[-1]['loss'],trainer_mean_train_loss=summaries[-1]['train_loss'] if summaries else None,gradients_all_finite=True,max_logged_grad_norm=max(grads),rank0_peak_allocated_gib=max(x.get('peak_allocated_gib',0) for x in logs)),validation=metrics,all_results_remote=True,started_at=state['started_at'],completed_at=now())
 (OUT/'report.json').write_text(json.dumps(report,indent=2));(RUN/'completed.json').write_text(json.dumps(report,indent=2));save('completed',child_pid=None,report=str(OUT/'report.json'))
except Exception as error:
 save('failed',error=str(error));(RUN/'failure.json').write_text(json.dumps(dict(error=str(error),time=now(),traceback=traceback.format_exc()),indent=2));traceback.print_exc();raise
