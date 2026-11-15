"""Finite after-training queue, with no recurring desktop automation."""
import fcntl
import os
import subprocess
import time
import traceback
from .common import *
from .prepare import bind_final_weights
from experiments.recover_validation_20261114.common import host_sample, safe
from experiments.recover_validation_20261114.controller import idle_gpus


def main():
    lock = (ROOT/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'controller.pid').exists(), 'Inspect existing controller before any recovery'
    (ROOT/'controller.pid').write_text(str(os.getpid()))
    verify_code(); assert read(ROOT/'preparation.json')['status'] == 'passed'
    while True:
        training = read(TRAIN/'status.json')
        assert training['status'] != 'failed', 'Training failed; do not use incomplete weights'
        if training['status'] == 'completed': break
        save(ROOT/'status.json', dict(status='waiting_for_all_five_final_checkpoints',
            completed_training=training.get('completed_models', []), updated=time.time(),
            controller_pid=os.getpid(), evaluation_started=False, desktop_automation_created=False))
        time.sleep(10)
    bind_final_weights(); manifest = read(ROOT/'inputs.json')
    pending = list(MODELS); active = {}; finished = {}; stable = {}; last_launch = 0
    initial_oom = host_sample()['oom_kill']
    while pending or active:
        for name, (child, gpu) in list(active.items()):
            if child.poll() is not None:
                path = ROOT/'jobs'/(name+'.json'); state = read(path) if path.exists() else {}
                finished[name] = dict(returncode=child.returncode, status=state.get('status','failed_before_status'), gpu=gpu)
                del active[name]
        resource = host_sample(); free = set(idle_gpus())-{g for p,g in active.values()}
        for g in range(4):
            if g in free and safe(resource,g,initial_oom): stable.setdefault(g,time.time())
            else: stable.pop(g,None)
        ready = sorted(g for g,t in stable.items() if time.time()-t >= 15)
        if pending and ready and time.time()-last_launch >= 5:
            name = pending.pop(0); gpu = ready[0]; info = manifest['model_checkpoints'][name]
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
                PYTORCH_ALLOC_CONF='expandable_segments:True',TRITON_CACHE_DIR=str(ROOT/'triton-cache'/name))
            for k in ('FOREST_NATIVE_GATED_NORM','PRM_CUDA_MEMORY_FRACTION'): env.pop(k,None)
            jobs = ROOT/'jobs'; jobs.mkdir(exist_ok=True)
            command = [info['python'],'-m',PACKAGE+'.worker',name]
            with (jobs/(name+'.log')).open('x') as log:
                child = subprocess.Popen(command,cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(jobs/(name+'-launch.json'),dict(pid=child.pid,gpu=gpu,command=command,started=time.time()))
            active[name] = (child,gpu);last_launch = time.time();stable.pop(gpu,None)
        save(ROOT/'status.json',dict(status='running',pending=pending,
            active={n:dict(pid=p.pid,gpu=g) for n,(p,g) in active.items()},finished=finished,
            updated=time.time(),controller_pid=os.getpid(),evaluation_started=True,desktop_automation_created=False))
        save(ROOT/'resources.json',resource)
        if pending or active: time.sleep(5)
    assert all(v['status']=='completed' and v['returncode']==0 for v in finished.values()), finished
    with (ROOT/'report.log').open('w') as log:
        result = subprocess.run([REPORT_PYTHON,'-m',PACKAGE+'.report'],cwd=ROOT/'code',
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode == 0, 'See report.log; preserve all predictions'
    report = read(OUT/'report.json'); assert report['status'] == 'completed'
    save(ROOT/'status.json',dict(status='completed',finished=finished,updated=time.time(),
        report=str(OUT/'report.json'),figure=report['figures'],desktop_automation_created=False))


if __name__ == '__main__':
    try: main()
    except BaseException:
        p=ROOT/'status.json';s=read(p) if p.exists() else {}
        s.update(status='failed',error=traceback.format_exc(),updated=time.time());save(p,s);raise
