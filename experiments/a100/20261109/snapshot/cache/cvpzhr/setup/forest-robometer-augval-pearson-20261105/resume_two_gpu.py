"""Resume the authorized Robometer evaluation on freed GPUs 0 and 2."""
import fcntl, os, signal, subprocess, sys, time, traceback
from common import *
from aggregate import validate
from controller import environment

ROBOMME = Path('/cache/robot-baselines/eval-results/robomme-closed-loop-20261105/summary.json')
GPU_SHARDS = {0: [0, 2], 2: [1, 3]}

def verify():
    p=plan()
    check_inputs(p)
    assert read(ROOT/'cpu-tests.json')['status']=='passed'
    assert read(ROOT/'input-checks.json')['status']=='passed'
    for path,digest in read(ROOT/'implementation-sha256.json').items():
        assert sha(path)==digest, path
    report=read(R_OUT/'report.json')
    assert report['status']=='completed' and report['epochs']==1 and report['steps']==9967
    receipt=read(R_CKPT/'complete.json')
    assert receipt['complete'] and receipt['step']==9967 and abs(receipt['epoch']-1)<1e-6
    assert read(R_CKPT/'validation.json')['eval_samples']==66891
    for name,digest in read(OUT/'checkpoints.json')['robometer']['files'].items():
        assert sha(R_CKPT/name)==digest,name
    records=validate([r for rank in range(4) for r in rows(OUT/'forest'/f'predictions-{rank}.jsonl')],p)
    counts={rank:sum(len(rows(e['timeline'])) for e in assigned(p,rank)) for rank in range(4)}
    assert sorted(rank for ranks in GPU_SHARDS.values() for rank in ranks)==list(range(4))
    assert sum(counts.values())==len(records)==78554
    mme=read(ROBOMME)
    assert mme['completed']==mme['target']==160, 'RoboMME evaluation must finish first'
    return dict(checked_at=now(),forest_samples=len(records),shard_samples=counts,
                gpu_shards=GPU_SHARDS,robomme_completed=mme['completed'],batch_size=8,
                worker_and_protocol_unchanged=True)

def selected_apps():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    selected={uuid.strip() for line in raw.splitlines() for index,uuid in [line.split(',')]
              if int(index) in GPU_SHARDS}
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return [line.strip() for line in raw.splitlines() if line.split(',')[0].strip() in selected]

def main():
    if '--check' in sys.argv:
        print(json.dumps(dict(**verify(),selected_gpu_apps=selected_apps()),ensure_ascii=False))
        return
    lock=(ROOT/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=now()
    active={}
    pending={gpu:list(ranks) for gpu,ranks in GPU_SHARDS.items()}
    def state(phase,**kw):
        write(ROOT/'status.json',dict(phase=phase,pid=os.getpid(),started_at=started,updated_at=now(),
             metrics=str(OUT/'metrics.json'),gpus=list(GPU_SHARDS),scheduler='two-gpu-resume',
             gpu_shards=GPU_SHARDS,**kw))
    def interrupted(sig,frame):
        raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        state('checking_resume')
        write(ROOT/'two-gpu-resume'/'preflight.json',verify())
        while True:
            apps=selected_apps()
            if not apps:break
            state('waiting_for_gpu_release',next_phase='robometer',compute_apps=apps)
            time.sleep(15)
        state('robometer')
        while active or any(pending.values()):
            for gpu,(rank,proc) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                if code!=0:raise RuntimeError(f'Robometer shard {rank} on GPU {gpu} exited {code}')
                del active[gpu]
            for gpu,ranks in pending.items():
                if gpu in active or not ranks:continue
                rank=ranks.pop(0)
                with (ROOT/f'robometer-{rank}.log').open('ab',buffering=0) as log:
                    proc=subprocess.Popen([R_PY,'-u',str(ROOT/'worker.py'),'robometer','--rank',str(rank)],
                        cwd=ROOT,env=environment('robometer',gpu),stdout=log,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,start_new_session=True)
                active[gpu]=(rank,proc)
            progress={f'robometer-{rank}':read(ROOT/f'robometer-{rank}.json')
                      for rank in range(4) if (ROOT/f'robometer-{rank}.json').exists()}
            state('robometer',workers=progress,
                  active={str(gpu):dict(rank=rank,pid=proc.pid) for gpu,(rank,proc) in active.items()},
                  pending=pending)
            if active:time.sleep(15)
        state('aggregating')
        with (ROOT/'aggregation.log').open('ab',buffering=0) as log:
            subprocess.run([R_PY,str(ROOT/'aggregate.py')],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(OUT/'metrics.json')['status']=='completed'
        state('completed',comparison=str(OUT/'comparison.md'))
    except BaseException as exc:
        for rank,proc in active.values():
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for rank,proc in active.values():
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        state('failed',error=repr(exc),traceback=traceback.format_exc())
        raise

if __name__=='__main__':main()
