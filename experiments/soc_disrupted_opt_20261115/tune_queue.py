"""One measured suite/checkpoint per idle GPU, separate from paused formal queue."""
import argparse
import fcntl
import os
import subprocess
import time
from .common import *
from experiments.soc_disrupted_eval_20261115.guard import host_sample, gpu_processes, idle_safe, process_identity, alive


def main(allowed_gpus=(0,1,2,3)):
    OPT.mkdir(parents=True,exist_ok=True);(OPT/'tuning').mkdir(exist_ok=True)
    lock=(OPT/'tuning.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=read(OPT/'tuning-status.json') if (OPT/'tuning-status.json').exists() else {}
    assert not alive(previous.get('controller'))
    for value in previous.get('active',{}).values():assert not alive(value['process'])
    assert all(not gpu_processes()[g] for g in allowed_gpus),'Tuning GPU occupied; inspect instead of competing'
    source=read(ROOT/'inputs.json');verify_contract(source)
    tasks=[f'{m}-{s}' for s in SUITES for m in MODELS]
    completed=[k for k in tasks if (OPT/'tuning'/f'{k}.json').exists()
               and read(OPT/'tuning'/f'{k}.json')['status']=='passed']
    active={};failures={};initial_oom=host_sample()['oom_kill'];last_launch=0
    stable={i:None for i in range(4)}
    (OPT/'tuning-controller.pid').write_text(str(os.getpid()))
    while True:
        now=time.time();sample=host_sample();processes=gpu_processes()
        for key,(p,g) in list(active.items()):
            if p.poll() is not None:
                del active[key];stable[g]=None
                path=OPT/'tuning'/f'{key}.json';result=read(path) if path.exists() else {}
                if p.returncode==0 and result.get('status')=='passed':completed.append(key)
                else:failures[key]=dict(exit_code=p.returncode,receipt=str(path))
        status=dict(status='running',controller=process_identity(os.getpid()),updated=now,
            active={k:dict(process=process_identity(p.pid),gpu=g) for k,(p,g) in active.items()},
            completed=completed,failures=failures,pending=[k for k in tasks if k not in completed and k not in active],resources=sample)
        if not active and (failures or len(completed)==len(tasks)):
            status['status']='failed' if failures else 'completed'
            save(OPT/'tuning-status.json',status);return
        save(OPT/'tuning-status.json',status)
        for gpu in allowed_gpus:
            if gpu in [g for p,g in active.values()] or not idle_safe(sample,processes,gpu,initial_oom):
                stable[gpu]=None;continue
            stable[gpu]=now if stable[gpu] is None else stable[gpu]
            if failures or now-stable[gpu]<60 or now-last_launch<60:continue
            pending=[k for k in tasks if k not in completed and k not in active]
            if not pending:break
            key=pending[0];info=source['registry'][key]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1',
                PYTHONPATH=f'{OPT}/code:{ROOT}/code')
            with (OPT/'tuning'/f'{key}.log').open('a') as log:
                p=subprocess.Popen([info['python'],'-m','experiments.soc_disrupted_opt_20261115.tune',
                    '--job',key,'--gpu',str(gpu),'--parent',str(os.getpid())],cwd=OPT/'code',env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=(p,gpu);last_launch=now;stable[gpu]=None
        time.sleep(5)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3');a=p.parse_args()
    main(tuple(map(int,a.gpus.split(','))))
