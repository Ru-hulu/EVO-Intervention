"""Build and verify caches, smoke-test fresh fine-tuning, then launch on GPUs 2,3."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/cache/cvpzhr/setup/forest-s01-decline-20261106')
PYTHON='/cache/cvpzhr/envs/forest/bin/python'
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'code'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')


def status(phase,**extra):
    path=ROOT/'status.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    state.update(phase=phase,controller_pid=os.getpid(),updated_at=time.time(),**extra)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(path)


def start(args,log,gpu):
    with (ROOT/log).open('a') as out:
        return subprocess.Popen([PYTHON,'-u',*args],cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES=gpu),
                                stdout=out,stderr=subprocess.STDOUT)


def main():
    import fcntl
    lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={int(line.split(',')[0]):int(line.split(',')[1]) for line in devices.splitlines()}
    if any(used[i]>1000 for i in [2,3]):
        status('gpu_conflict',memory_mib=used);return 1
    workers=[start(['prepare.py','worker','--worker',str(i),'--workers','8'],f'worker-{i}.log',str(2+i%2)) for i in range(8)]
    status('building_cache',worker_pids=[p.pid for p in workers],training_started=False)
    while any(p.poll() is None for p in workers):
        failed=[p for p in workers if p.poll() is not None and p.returncode!=0]
        if failed:
            for p in workers:
                if p.poll() is None:p.terminate()
            status('cache_failed',returncodes=[p.poll() for p in workers]);return 1
        time.sleep(10)
    if any(p.returncode for p in workers):
        status('cache_failed',returncodes=[p.returncode for p in workers]);return 1
    status('validating_cache')
    validation=start(['prepare.py','validate'],'cache-validation.log','')
    if validation.wait():status('validation_failed');return 1
    config=json.loads((ROOT/'config.json').read_text())
    smoke=dict(config,output_dir=str(ROOT/'smoke-output'),max_steps=2,validation_sample_files=[],num_workers=0,
               sample_files=[config['sample_files'][0]],log_every_steps=1)
    (ROOT/'smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
    command=['-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','mem_lite.train_stage3','--config']
    status('smoke_test')
    smoke_process=start(command+[str(ROOT/'smoke.json')],'smoke.log','2,3')
    if smoke_process.wait():status('smoke_failed');return 1
    assert not Path(config['output_dir']).exists(), 'Refuse to append to a prior run'
    training=start(command+[str(ROOT/'config.json')],'train.log','2,3')
    status('training_launch',training_pid=training.pid,training_started=False,output_dir=config['output_dir'])
    while training.poll() is None:
        log=Path(config['output_dir'])/'train.jsonl'
        if log.exists() and log.stat().st_size:
            last=json.loads(log.read_text().splitlines()[-1])
            status('training',training_started=True,last_step=last['step'],last_loss=last['loss'])
        time.sleep(15)
    status('completed' if training.returncode==0 else 'training_failed',returncode=training.returncode)
    return training.returncode


if __name__=='__main__':
    sys.exit(main())
