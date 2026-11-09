from pathlib import Path
import json,os,subprocess,time,traceback,fcntl
RUN=Path('/cache/cvpzhr/setup/robometer-20261031');CODE=Path('/cache/cvpzhr/code/robometer-proprimem-20261031');PY='/cache/cvpzhr/envs/robometer/bin/python';DATA=Path('/cache/cvpzhr/datasets/robometer-proprimem-5fps-w8-20261031')
lock=(RUN/'checks.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ENV={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0,1,2,3','PYTHONNOUSERSITE':'1'}
def status(phase,**kw):
 p=RUN/'checks-status.json.tmp';p.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw),indent=2));p.replace(RUN/'checks-status.json')
def idle():
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
 return not procs and all(int(line.split(',')[0])<=5 and int(line.split(',')[1])<1024 for line in values.strip().splitlines())
def wait_idle():
 status('waiting_for_idle_gpu')
 while True:
  if idle():
   time.sleep(15)
   if idle():return
  time.sleep(15)
def run(args,log,timeout=900):
 with (RUN/log).open('a') as f:
  p=subprocess.run([PY,*args],cwd=CODE,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 return p.returncode
try:
 status('waiting_for_model')
 while True:
  state=json.loads((RUN/'download-status.json').read_text())
  if state['phase']=='failed':raise RuntimeError('Model download failed')
  if state['phase']=='ready':break
  time.sleep(10)
 wait_idle();status('gpu_consistency_check')
 if run([str(RUN/'gpu_check.py')],'gpu-check.log')!=0:raise RuntimeError('GPU consistency check failed; see gpu-check.log')
 status('waiting_for_cache')
 while True:
  state=json.loads((DATA/'status.json').read_text())
  if state['phase']=='failed':raise RuntimeError('Cache preparation failed')
  if state['phase']=='completed':break
  time.sleep(10)
 if run([str(RUN/'prepare_benchmark.py')],'prepare-benchmark.log')!=0:raise RuntimeError('Cache audit failed')
 results=[]
 for batch in (4,8,16,32):
  wait_idle();status('benchmarking',per_gpu_batch=batch,completed=results)
  output=RUN/f'batch-{batch}.json';log=f'batch-{batch}.log'
  args=['-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(RUN/'benchmark.py'),'--config',str(RUN/'config.json'),'--batch',str(batch),'--output',str(output)]
  rc=run(args,log)
  if rc:
   text=(RUN/log).read_text()
   if 'out of memory' not in text.lower():raise RuntimeError(f'Batch {batch} failed; see {log}')
   failure=dict(status='oom',per_gpu_batch=batch,gradient_checkpointing=False);results.append(failure);output.write_text(json.dumps(failure,indent=2))
   if batch<32:break
   wait_idle();status('benchmarking',per_gpu_batch=32,gradient_checkpointing=True,completed=results)
   out=RUN/'batch-32-gc.json';rc=run([*args[:-1],str(out),'--checkpointing'],'batch-32-gc.log')
   if rc:raise RuntimeError('Batch32 checkpointed benchmark failed')
   results.append(json.loads(out.read_text()))
  else:results.append(json.loads(output.read_text()))
 passed=[r for r in results if r['status']=='passed' and r['peak_reserved_gib']<74]
 if not passed:raise RuntimeError('No batch with safe memory headroom')
 best=max(passed,key=lambda x:x['samples_per_second'])
 config=json.loads((RUN/'config.json').read_text());config['training'].update(per_device_train_batch_size=best['per_gpu_batch'],gradient_checkpointing=best['gradient_checkpointing'])
 (RUN/'recommended-config.json').write_text(json.dumps(config,indent=2))
 result=dict(status='READY_FOR_LORA_TRAINING_NOT_STARTED',benchmarks=results,recommended=best,cache=str(DATA),environment=PY,model='/cache/cvpzhr/models/Robometer-4B',formal_training_started=False,optimizer_steps=0)
 (RUN/'readiness.json').write_text(json.dumps(result,indent=2));status('completed',recommended=best)
except Exception as error:
 status('failed',error=str(error));traceback.print_exc();raise
