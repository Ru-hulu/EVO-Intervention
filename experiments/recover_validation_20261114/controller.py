"""Finite remote evaluation queue, one worker per verified idle GPU, no desktop timer."""
import fcntl
import argparse
import os
import subprocess
import time
from .common import *


def idle_gpus():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    contexts=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return [int(i.strip()) for i,u,m in (line.split(',') for line in raw.splitlines())
            if u.strip() not in contexts and float(m)<1000]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adopt-repair',type=Path);args=parser.parse_args()
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.adopt_repair:assert not (ROOT/'controller.pid').exists(), 'Existing launch requires explicit recovery audit'
    else:
        repair=read(args.adopt_repair)
        assert repair['reason']=='forest_constant_lr_verification_only' and repair['failed_model']=='forest'
        for pid in (repair['old_controller_pid'],repair['failed_worker_pid']):assert not alive(pid)
    (ROOT/'controller.pid').write_text(str(os.getpid()))
    jobs=ROOT/'jobs';jobs.mkdir(exist_ok=True);manifest=read(ROOT/'inputs.json')
    assert manifest['models']==list(MODELS)
    pending=list(MODELS);active={};finished={};stable={};last_launch=0
    baseline=host_sample();initial_oom=baseline['oom_kill']
    if args.adopt_repair:
        previous=repair['previous_status'];pending=previous['pending']+['forest']
        finished={k:v for k,v in previous['finished'].items() if k!='forest'}
        assert 'forest' not in previous['active'] and 'forest' not in previous['pending']
        for name,value in previous['active'].items():
            launch=read(ROOT/'jobs'/(name+'-launch.json'));assert launch['pid']==value['pid'] and launch['gpu']==value['gpu']
            assert alive(value['pid']) and identity(value['pid'])==repair['active_identities'][name]
            active[name]=(Adopted(value['pid'],name),value['gpu'])
    while pending or active:
        for name,(p,gpu) in list(active.items()):
            if p.poll() is not None:
                state=read(jobs/(name+'.json')) if (jobs/(name+'.json')).exists() else {}
                finished[name]=dict(returncode=p.returncode,status=state.get('status','failed_before_status'),gpu=gpu)
                del active[name]
        r=host_sample();free=set(idle_gpus())-{gpu for p,gpu in active.values()}
        for gpu in range(4):
            if gpu in free and safe(r,gpu,initial_oom):stable.setdefault(gpu,time.time())
            else:stable.pop(gpu,None)
        ready=[g for g,t in stable.items() if time.time()-t>=30]
        if pending and ready and time.time()-last_launch>=30:
            name=pending.pop(0);gpu=ready[0];info=manifest['model_checkpoints'][name]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
                PYTORCH_ALLOC_CONF='expandable_segments:True')
            for key in ('PRM_CUDA_MEMORY_FRACTION','FOREST_NATIVE_GATED_NORM'):env.pop(key,None)
            command=[info['python'],'-m','experiments.recover_validation_20261114.worker',name]
            with (jobs/(name+'.log')).open('a') as log:
                p=subprocess.Popen(command,cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(jobs/(name+'-launch.json'),dict(pid=p.pid,gpu=gpu,command=command,started=time.time()))
            active[name]=(p,gpu);last_launch=time.time();stable.pop(gpu,None)
        save(ROOT/'status.json',dict(status='running',pending=pending,active={n:dict(pid=p.pid,gpu=g) for n,(p,g) in active.items()},
            finished=finished,updated=time.time(),desktop_automation_created=False))
        save(ROOT/'resources.json',r)
        if pending or active:time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    report=subprocess.run([PYTHON,'-m','experiments.recover_validation_20261114.report'],cwd=ROOT/'code',env=env)
    save(ROOT/'status.json',dict(status='predictions_completed' if all(v['status']=='completed' for v in finished.values())
        else 'needs_inspection',finished=finished,report_returncode=report.returncode,updated=time.time(),desktop_automation_created=False))


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'


def identity(pid):
    return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]


class Adopted:
    def __init__(self,pid,name):self.pid,self.name=pid,name
    def poll(self):
        if alive(self.pid):return None
        state=read(ROOT/'jobs'/(self.name+'.json'))
        return 0 if state['status'] in ('completed','paused_for_resources') else 1


if __name__=='__main__':main()
