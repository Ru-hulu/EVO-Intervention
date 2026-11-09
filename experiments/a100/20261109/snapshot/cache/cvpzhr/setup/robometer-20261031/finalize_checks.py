from pathlib import Path
import json,os,time,subprocess,traceback,fcntl
RUN=Path('/cache/cvpzhr/setup/robometer-20261031');CODE=Path('/cache/cvpzhr/code/robometer-proprimem-20261031');PY='/cache/cvpzhr/envs/robometer/bin/python'
ENV={**os.environ,'PYTHONPATH':str(CODE)+':'+str(CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0,1,2,3','PYTHONNOUSERSITE':'1'}
lock=(RUN/'finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(phase,**kw):
 p=RUN/'finalize-status.json.tmp';p.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw),indent=2));p.replace(RUN/'finalize-status.json')
def idle():
 procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
 return not procs and all(int(x.split(',')[0])<=5 and int(x.split(',')[1])<1024 for x in values.strip().splitlines())
try:
 status('waiting_for_initial_benchmarks')
 while True:
  s=json.loads((RUN/'checks-status.json').read_text())
  if s['phase']=='failed':raise RuntimeError(s.get('error'))
  if s['phase']=='completed':break
  time.sleep(5)
 result=json.loads((RUN/'readiness.json').read_text())
 if any(r['status']=='oom' and r['per_gpu_batch']<32 for r in result['benchmarks']):
  status('waiting_for_idle_gpu')
  while True:
   if idle():
    time.sleep(15)
    if idle():break
   time.sleep(15)
  status('benchmarking_32_with_checkpointing')
  args=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(RUN/'benchmark.py'),'--config',str(RUN/'config.json'),'--batch','32','--checkpointing','--output',str(RUN/'batch-32-gc.json')]
  with (RUN/'batch-32-gc.log').open('a') as log:
   rc=subprocess.run(args,cwd=CODE,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
  if rc:raise RuntimeError('Checkpointed batch32 benchmark failed; see batch-32-gc.log')
  result['benchmarks'].append(json.loads((RUN/'batch-32-gc.json').read_text()))
 safe=[r for r in result['benchmarks'] if r['status']=='passed' and r['peak_reserved_gib']<74]
 best=max(safe,key=lambda x:x['samples_per_second']);result['recommended']=best
 config=json.loads((RUN/'config.json').read_text());config['training'].update(per_device_train_batch_size=best['per_gpu_batch'],gradient_checkpointing=best['gradient_checkpointing'])
 (RUN/'recommended-config.json').write_text(json.dumps(config,indent=2))
 status('checking_training_entry')
 with (RUN/'training-entry-check.log').open('a') as log:
  rc=subprocess.run([PY,str(RUN/'check_entry.py')],cwd=CODE,env={**ENV,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
 if rc:raise RuntimeError('Training entry configuration check failed')
 assert json.loads((RUN/'mixed-input-check.json').read_text())['status']=='passed'
 assert json.loads((RUN/'gpu-check.json').read_text())['status']=='passed'
 result.update(data_check=json.loads((RUN/'data-check.json').read_text()),gpu_check=json.loads((RUN/'gpu-check.json').read_text()),input_check=json.loads((RUN/'mixed-input-check.json').read_text()),training_entry_check=json.loads((RUN/'training-entry-check.json').read_text()),benchmark_scope='Four-GPU DDP; measured data loading + forward/backward + gradient clipping. AdamW moment memory reserved; no optimizer steps.',batch_search_finished=True)
 (RUN/'readiness.json').write_text(json.dumps(result,indent=2));status('completed',recommended=best)
except Exception as error:status('failed',error=str(error));traceback.print_exc();raise
