"""A checkpoint owns one GPU pair from its tuning through its audited evaluation."""
import argparse
import fcntl
import os
import subprocess
import time
import traceback
from .common import *
from .guard import process_identity, alive, gpu_processes, host_sample, safe


def run():
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--gpus',required=True);a=p.parse_args()
    task=a.task;gpus=[int(x) for x in a.gpus.split(',')];assert len(set(gpus))==2
    taskdir=ROOT/'tasks'/task;taskdir.mkdir(parents=True,exist_ok=True)
    lock=(taskdir/'manager.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data,execution=source();info=data['registry'][task];verify_code();entries=streams(data)
    rows=[r for r in data['episodes'] if r['suite']==info['suite']]
    assert not (taskdir/'choice.json').exists(), 'Completed tuning must not be overwritten'
    assert all(not gpu_processes()[g] for g in gpus)
    manager=process_identity(os.getpid());initial=host_sample();children={}
    state=dict(manager=manager,task=task,gpus=gpus,started=time.time(),status='loading_replicas')
    def update(**kw):
        state.update(kw,updated=time.time(),workers={str(k):dict(pid=v.pid) for k,v in children.items()})
        save(taskdir/'status.json',state)
    def check():
        update()
        assert alive(read(ROOT/'status.json')['controller']), 'Queue controller exited'
        assert not (ROOT/'pause-all').exists(), 'Queue pause requested'
        for rank,child in children.items():
            if child.poll() is not None and child.returncode!=0:
                raise RuntimeError(f'Worker {rank} exited {child.returncode}; inspect its log')
        assert all(safe(host_sample(),g,initial['oom_kill']) for g in gpus), 'Resource protection'
    def wait(predicate,timeout=1800):
        began=time.time()
        while not predicate():
            check();assert time.time()-began<timeout,'Worker/barrier timeout';time.sleep(2)
        check()
    update()
    try:
        # Two replicas load at a time, one per GPU, to bound transient CPU memory.
        for replica in range(4):
            for side,gpu in enumerate(gpus):
                rank=replica*2+side;folder=taskdir/'workers'/str(rank);folder.mkdir(parents=True,exist_ok=True)
                ticket=folder/'ticket.json'
                save(ticket,dict(task=task,rank=rank,replica=replica,gpu=gpu,manager=manager,
                    initial_oom=initial['oom_kill'],execution_sha256=sha(ROOT/'execution.json')))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                    TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
                    TRITON_CACHE_DIR=str(ORIGINAL/'triton'/('forest-Order-part0' if info['model']=='forest' else task)))
                for k in ('PYTHONPATH','FOREST_NATIVE_GATED_NORM','PRM_CUDA_MEMORY_FRACTION','PYTORCH_CUDA_ALLOC_CONF'):env.pop(k,None)
                with (folder/'worker.log').open('w') as log:
                    children[rank]=subprocess.Popen([info['python'],'-m',PACKAGE+'.worker','--ticket',str(ticket)],
                        cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            wait(lambda:all((taskdir/'workers'/str(replica*2+s)/'loaded.json').exists() for s in (0,1)))
        proofs=[read(taskdir/'workers'/str(s)/'equivalence.json') for s in (0,1)]
        audits=[read(taskdir/'workers'/str(r)/'loaded.json')['initialization'] for r in range(8)]
        assert all(v==audits[0] for v in audits), 'Replica initialization differs'
        eligible=set(proofs[0]['eligible_batches'])&set(proofs[1]['eligible_batches']);assert eligible
        if info['model']=='procvlm-2b':
            def rate(batch):return sum(next(r['observations_per_second'] for r in q['trials'] if r['batch_size']==batch) for q in proofs)
            batch=max(eligible,key=rate)
        else:batch=1
        save(taskdir/'batch.json',dict(batch_size=batch,equivalence=proofs,initialization=audits[0]))
        table=[]
        for count in (1,2,3,4):
            stage=taskdir/'tuning'/str(count);stage.mkdir(parents=True,exist_ok=True)
            ranks=list(range(2*count));update(status='tuning',replicas_per_gpu=count,batch_size=batch)
            save(stage/'prepare.json',dict(replicas_per_gpu=count,started=time.time()))
            wait(lambda:all((stage/f'ready-{r}.json').exists() for r in ranks))
            save(stage/'go.json',dict(seconds=40,started=time.time()))
            samples=[]
            while not all((stage/f'result-{r}.json').exists() for r in ranks):
                check();samples.append(host_sample());time.sleep(2)
            results=[read(stage/f'result-{r}.json') for r in ranks]
            assert all(r['status']=='passed' for r in results)
            trial=dict(replicas_per_gpu=count,batch_size=batch,passed=True,
                observations_per_second=sum(r['observations_per_second'] for r in results),
                sum_replica_peak_reserved_gib=sum(r['peak_reserved_gib'] for r in results),
                max_used_gib=max(s['gpus'][g]['used_mib']/1024 for s in samples for g in gpus),
                max_temperature=max(s['gpus'][g]['temperature'] for s in samples for g in gpus),results=results)
            table.append(trial);save(stage/'resources.json',samples);save(stage/'done.json',trial)
        choice=choose(table)
        choice=dict(choice,status='passed',task=task,checkpoint_sha256=info['checkpoint_sha256'],
            inputs_sha256=execution['inputs_sha256'],gpus=gpus,trials=table,
            selection_rule='smallest replica count within 3% of measured maximum; only safe/equivalent candidates',
            batch_sha256=sha(taskdir/'batch.json'),finished=time.time())
        initialize_queue(taskdir/'queue.sqlite',[(r['key'],level,len(entries[level][r['key']]['target_pt']))
            for r in rows for level in range(2,6)])
        save(taskdir/'choice.json',choice)
        inactive=[r for r in children if r//2>=choice['replicas_per_gpu']]
        wait(lambda:all(children[r].poll() is not None for r in inactive))
        update(status='evaluating',choice=choice)
        save(taskdir/'formal-go.json',dict(started=time.time(),inputs_sha256=execution['inputs_sha256']))
        while any(child.poll() is None for child in children.values()):
            check();time.sleep(5)
        assert all(child.returncode==0 for child in children.values())
        c=db(taskdir/'queue.sqlite');counts=dict(c.execute('SELECT status,count(*) FROM jobs GROUP BY status').fetchall());c.close()
        assert counts=={'completed':4*len(rows)},counts
        from .report import task_report
        report=task_report(task);assert report['complete']
        save(taskdir/'verification.json',report);update(status='completed',finished=time.time());return 0
    except BaseException:
        error=traceback.format_exc();(taskdir/'abort').write_text(error)
        update(status='stopping_after_error',error=error)
        # Only our children cooperate and save prefixes. Other work is untouched.
        while any(child.poll() is None for child in children.values()):update();time.sleep(2)
        update(status='failed',error=error,finished=time.time());return 1


if __name__=='__main__':raise SystemExit(run())
