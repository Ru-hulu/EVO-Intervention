import fcntl
import os
import subprocess
import time
import traceback
from .common import *
from .guard import alive, process_identity, gpu_processes, host_sample, safe


def run():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'status.json').exists():assert not alive(read(ROOT/'status.json').get('controller'))
    assert not any(alive(read(p).get('process')) for p in ROOT.glob('tasks/*/workers/*/status.json'))
    verify_code();data,execution=source()
    assert read(ROOT/'preflight.json')['status']=='passed'
    initial=host_sample();controller=process_identity(os.getpid());(ROOT/'controller.pid').write_text(str(os.getpid())+'\n')
    # Long Order tasks first; two independently tuning/evaluating checkpoint groups.
    pending=[f'{m}-{s}' for s in SUITES for m in MODELS]
    done=[];failures=[];active={};stable={}
    state=dict(status='starting',controller=controller,started=time.time(),inputs_sha256=execution['inputs_sha256'])
    def update(**kw):
        state.update(kw,updated=time.time(),pending=pending,completed=done,failures=failures,
            active={str(k):dict(task=v['task'],gpus=v['gpus'],pid=v['child'].pid) for k,v in active.items()})
        save(ROOT/'status.json',state)
    update()
    while pending or active:
        for pair,v in list(active.items()):
            code=v['child'].poll()
            if code is None:continue
            p=ROOT/'tasks'/v['task']/'verification.json'
            if code==0 and p.exists() and read(p)['complete']:done.append(v['task'])
            else:failures.append(dict(task=v['task'],returncode=code,log=v['log']))
            del active[pair]
        sample=host_sample();processes=gpu_processes();now=time.time()
        for pair,gpus in enumerate(((0,1),(2,3))):
            if pair in active or not pending or (ROOT/'pause-all').exists():continue
            ready=all(not processes[g] and safe(sample,g,initial['oom_kill']) for g in gpus)
            if ready:stable[pair]=stable.get(pair,now)
            else:stable.pop(pair,None)
            if not ready or now-stable[pair]<30:continue
            task=pending.pop(0);folder=ROOT/'tasks'/task;folder.mkdir(parents=True,exist_ok=True)
            log=folder/'manager.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            with log.open('w') as f:
                child=subprocess.Popen([PYTHON,'-m',PACKAGE+'.pair','--task',task,'--gpus',','.join(map(str,gpus))],
                    cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            active[pair]=dict(task=task,gpus=list(gpus),child=child,log=str(log));stable.pop(pair,None)
        update(status='running' if active else 'waiting_for_safe_gpu_pair')
        if (ROOT/'pause-all').exists() and not active:update(status='paused');return
        if pending or active:time.sleep(5)
    from .report import report
    result=report();update(status='completed' if result['status']=='completed' and not failures else 'completed_with_failures',
                         report=str(OUT/'metrics.json'),finished=time.time())


if __name__=='__main__':
    try:run()
    except BlockingIOError:raise SystemExit('A controller already owns this queue')
    except BaseException:
        s=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        s.update(status='failed',error=traceback.format_exc(),updated=time.time());save(ROOT/'status.json',s);raise
