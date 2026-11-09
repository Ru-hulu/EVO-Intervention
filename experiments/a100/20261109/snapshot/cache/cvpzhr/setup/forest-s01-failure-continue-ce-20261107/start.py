"""Wait for verified data, build only new caches, then start authorized training."""
import json,os,time,subprocess
from pathlib import Path
import psutil
WORK=Path('/cache/cvpzhr/setup/forest-s01-failure-continue-ce-20261107')
DATA=Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261107-v5-failure-continue')
BATCH=Path('/cache/robot-baselines/eval-results/S01-Failure-Continue-20261107-r2')
PYTHON='/cache/cvpzhr/envs/forest/bin/python'
def status(phase,**kwargs):
    p=WORK/'pipeline-status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(phase=phase,updated_at=time.time(),pid=os.getpid(),**kwargs),indent=2)+'\n');tmp.replace(p)
def gpu_free():
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    return all(int(used[i])<1000 for i in [2,3])
def main():
    import fcntl
    lock=(WORK/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    proof=DATA/'verification/history-pairs.json'
    while not proof.exists():
        states=[json.loads(p.read_text()) for p in (BATCH/'status').glob('*.json')]
        assert all(s['returncode']==0 for s in states), 'Collection failure: inspect batch logs'
        try:
            proc=psutil.Process(1319802)
            alive=proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE and 's01_failure_continue/finish_batch.py' in ' '.join(proc.cmdline())
        except psutil.NoSuchProcess:alive=False
        assert alive, 'Dataset processing stopped without verification; inspect finish log'
        status('waiting_for_verified_dataset',collected=len(states),target=24)
        time.sleep(15)
    assert json.loads(proof.read_text())['complete']
    while not gpu_free():
        status('waiting_for_gpu_2_3');time.sleep(15)
    env=dict(os.environ,PYTHONPATH=str(WORK/'code'),PYTHONUTF8='1',OMP_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    status('preparing')
    subprocess.run([PYTHON,str(WORK/'prepare.py')],check=True,env=env)
    status('building_new_caches')
    jobs=[]
    for slot,gpu in enumerate([2,3]):
        with (WORK/f'cache-{gpu}.log').open('w') as log:
            jobs.append(subprocess.Popen([PYTHON,'-u',str(WORK/'cache.py'),'worker','--worker',str(slot),'--workers','2'],
                env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs];assert codes==[0,0],codes
    status('validating_caches')
    subprocess.run([PYTHON,str(WORK/'cache.py'),'validate'],env=env,check=True)
    status('training_controller')
    subprocess.run([PYTHON,str(WORK/'launch.py')],env=env,check=True)
    status('completed')
if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));raise
