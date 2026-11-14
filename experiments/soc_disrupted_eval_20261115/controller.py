"""Explicit opt-in isolated queue. Never touch existing training/evaluation jobs."""
import argparse
import fcntl
import os
import secrets
import subprocess
import time
from .common import *
from .guard import process_identity, alive, gpu_processes, idle_safe, host_sample


def queue(source):
    canaries = {f'canary-{m}-{s}':dict(model=m,suite=s,phase='canary') for m in MODELS for s in SUITES}
    return canaries | jobs(source)


def eligible(job, statuses, input_sha):
    if job['phase']=='canary':
        return True
    # All nine suite-specific GPU canaries must pass before formal inference.
    return all(statuses.get(f'canary-{m}-{s}',{}).get('status')=='completed' and
               statuses[f'canary-{m}-{s}']['inputs_sha256']==input_sha for m in MODELS for s in SUITES)


def run(resume=False):
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for directory in ('jobs','tickets','canary','history'):
        (ROOT/directory).mkdir(exist_ok=True)
    old=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
    assert not alive(old.get('controller')), 'Existing controller still alive'
    launches=read(ROOT/'launches.json') if (ROOT/'launches.json').exists() else []
    for launch in launches:
        assert not alive(launch.get('process')), 'Old worker is still alive; never duplicate it'
    previous={p.stem:read(p) for p in (ROOT/'jobs').glob('*.json') if not p.stem.endswith('-initialization')}
    for item in previous.values():
        assert not alive(item.get('process')), 'Old child still alive'
    assert not launches or resume, 'Use --resume explicitly after inspecting prior workers'
    assert all(v.get('status') in ('completed','paused') for v in previous.values()), 'Inspect failed/disappeared jobs before recovery'
    source=read(ROOT/'inputs.json'); verify_contract(source)
    input_sha=sha(ROOT/'inputs.json'); controller=process_identity(os.getpid())
    (ROOT/'controller.pid').write_text(str(os.getpid()))
    tasks=queue(source); active={}; initial=host_sample(); initial_oom=initial['oom_kill']
    if old:
        assert old.get('initial_oom',initial_oom)==initial_oom, 'New OOM since previous run requires inspection'
    statuses={k:v for k,v in previous.items() if k in tasks}
    stable={g:None for g in range(4)}; last_launch=0; attempts={k:[] for k in tasks}; failures={}
    while True:
        now=time.time(); sample=host_sample(); processes=gpu_processes()
        for key,(process,gpu) in list(active.items()):
            if process.poll() is not None:
                path=ROOT/'jobs'/f'{key}.json'
                state=read(path) if path.exists() else dict(status='failed',reason='Worker exited before initialization')
                statuses[key]=state; del active[key]; stable[gpu]=None
                if process.returncode not in (0,75) or state['status'] not in ('completed','paused'):
                    failures[key]=state
        state=dict(status='running' if not failures else 'failed_waiting_healthy_workers',controller=controller,
            inputs_sha256=input_sha, initial_oom=initial_oom, updated=now, evaluation_started=True,
            active={k:dict(pid=p.pid,gpu=g) for k,(p,g) in active.items()},
            completed=sum(v.get('status')=='completed' for v in statuses.values()), total_jobs=len(tasks),
            failures=failures, resources=sample)
        save(ROOT/'status.json',state)
        if failures and not active:
            state.update(status='failed',updated=time.time());save(ROOT/'status.json',state);return
        if all(statuses.get(k,{}).get('status')=='completed' for k in tasks) and not active:
            from .report import report
            result=report()
            assert result['status']=='completed'
            state.update(status='completed',updated=time.time());save(ROOT/'status.json',state);return
        if sample['oom_kill'] != initial_oom:
            failures['resource_guard']={'reason':'New OOM; inspect, never silently reset the OOM baseline'}
        for gpu in range(4):
            if gpu in [g for _,g in active.values()] or not idle_safe(sample,processes,gpu,initial_oom):
                stable[gpu]=None;continue
            stable[gpu]=now if stable[gpu] is None else stable[gpu]
            if failures or (ROOT/'pause-all').exists() or now-stable[gpu]<60 or now-last_launch<60:
                continue
            candidate=None
            for key,job in tasks.items():
                prior=statuses.get(key,{})
                if key in active or prior.get('status')=='completed' or not eligible(job,statuses,input_sha):continue
                recent=[t for t in attempts[key] if now-t<3600]
                if len(recent)>=3 or (recent and now-recent[-1]<120):continue
                candidate=(key,job);break
            if candidate is None:continue
            key,job=candidate; token=secrets.token_hex(24)
            ticket=dict(job_id=key,job=job,token=token,gpu=gpu,controller=controller,
                initial_oom=initial_oom,issued=time.time(),inputs_sha256=input_sha)
            save(ROOT/'tickets'/f'{key}.json',ticket)
            if key in statuses:
                save(ROOT/'history'/f'{key}-{time.time_ns()}.json',statuses[key])
            info=source['registry'][f"{job['model']}-{job['suite']}"]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONDONTWRITEBYTECODE='1')
            with (ROOT/'jobs'/f'{key}.log').open('a') as log:
                process=subprocess.Popen([info['python'],'-m','experiments.soc_disrupted_eval_20261115.worker',
                    '--job',key,'--ticket',token],cwd=Path(__file__).parents[2],env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=process_identity(process.pid)
            launches.append(dict(job_id=key,process=identity,gpu=gpu,started=now));save(ROOT/'launches.json',launches)
            active[key]=(process,gpu);attempts[key].append(now);last_launch=now;stable[gpu]=None
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start-evaluation',required=True,action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.resume)
