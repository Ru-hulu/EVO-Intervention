"""Wait for a genuinely idle GPU; never terminate other workloads."""
import subprocess,time,json,os,fcntl,traceback
from pathlib import Path
SETUP=Path('/cache/robot-baselines/setup/robomme-coin-20261105')
LOCKS=Path('/cache/robot-baselines/locks');LOCKS.mkdir(exist_ok=True)
lease=(LOCKS/'robomme-coin-20261105.lock').open('w')
fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(**kw):
 kw['time']=time.strftime('%Y-%m-%dT%H:%M:%S%z');p=SETUP/'status.tmp';p.write_text(json.dumps(kw,indent=2));p.replace(SETUP/'status.json')
try:
 assert (SETUP/'preflight.json').exists(),'Run and pass CPU preflight before enqueueing'
 streak={};status(stage='waiting_for_idle_gpu',criteria='memory < 1000 MiB and utilization <= 5%, 4 samples 30s apart')
 while True:
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
  readings=[];chosen=None
  for line in raw.strip().splitlines():
   gpu,mem,util=map(int,line.split(','));readings.append({'gpu':gpu,'memory_mib':mem,'utilization':util})
   streak[gpu]=streak.get(gpu,0)+1 if mem<1000 and util<=5 else 0
   if streak[gpu]>=4 and chosen is None:chosen=gpu
  if chosen is not None:
   gpu_lock=(LOCKS/f'gpu-{chosen}.lock').open('w')
   try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:streak[chosen]=0;time.sleep(30);continue
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(chosen)
   status(stage='extracting_history_features',gpu=chosen)
   (SETUP/'selected-gpu.json').write_text(json.dumps({'gpu':chosen,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')}))
   with (SETUP/'pipeline.log').open('a') as log:
    result=subprocess.run(['bash',str(SETUP/'launch.sh')],env=env,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:status(stage='failed',gpu=chosen,exit_code=result.returncode,log=str(SETUP/'pipeline.log'))
   break
  status(stage='waiting_for_idle_gpu',readings=readings,stable_idle_samples=streak)
  time.sleep(30)
except Exception:
 status(stage='failed',error=traceback.format_exc());raise
