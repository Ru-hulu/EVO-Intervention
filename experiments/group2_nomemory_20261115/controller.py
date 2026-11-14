"""Use four idle GPUs; all nine smoke tests must pass before formal evaluation."""
import fcntl
import os
import subprocess
import time
import traceback
from .common import *
from experiments.soc35_finetune_20261114.queue import host_health

def idle_gpus():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    return [int(i) for i,u,m in (tuple(x.strip() for x in line.split(',')) for line in raw.splitlines())
        if int(m)<1000 and u not in processes]

def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=read(ROOT/'inputs.json');registry=source['registry'];baseline=host_health()
    assert not baseline['under_oom'] and baseline['resident']<.8*baseline['limit']
    save(ROOT/'resource-baseline.json',baseline)
    state=dict(status='running',controller_pid=os.getpid(),started=time.time(),active={},completed=[],failed=[],
        conditions=['no_memory'],SA_success_policy=read(ROOT/'success-policy.json'),inputs_sha256=sha(ROOT/'inputs.json'))
    for phase in ['smoke','formal']:
        jobs={}
        for base,info in registry.items():
            choice=read(ROOT/'smoke'/base/'verification.json')['choice'] if phase=='formal' else None
            shards=4 if choice and choice['batch_size']==1 else 1
            rows=[r for r in source['episodes'] if r['suite']==info['suite']]
            for shard in range(shards):
                name=base+(f'-part{shard}' if shards>1 else '')
                seconds=(sum(len(r['frame_indices']) for r in rows[shard::shards])/choice['samples_per_second']
                    if choice else sum(len(r['frame_indices']) for r in rows)*(3 if info['model']=='procvlm-2b' else 1))
                jobs[name]=dict(base=base,info=info,shard=shard,shards=shards,estimated_seconds=seconds)
        save(ROOT/f'{phase}-jobs.json',{n:{k:v for k,v in j.items() if k!='info'} for n,j in jobs.items()})
        pending=sorted(jobs,key=lambda k:jobs[k]['estimated_seconds'],reverse=True)
        pending=[j for j in pending if not (ROOT/phase/j/'verification.json').exists()]
        active={};failures=[]
        while pending or active:
            for gpu,job in list(active.items()):
                child=job['process'];code=child.poll()
                if code is None:continue
                folder=ROOT/phase/job['name'];save(folder/f'exit-{child.pid}.json',dict(returncode=code,finished=time.time()))
                proof=folder/'verification.json'
                if code==0 and proof.exists() and read(proof)['status'] in ('passed','completed'):
                    state['completed'].append(dict(phase=phase,job=job['name']))
                else:failures.append(dict(phase=phase,job=job['name'],returncode=code,log=str(job['log'])))
                del active[gpu]
            for gpu in idle_gpus():
                if not pending or gpu in active or failures:continue
                name=pending.pop(0);job=jobs[name];info=job['info'];folder=ROOT/phase/name;folder.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                    TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
                    TRITON_CACHE_DIR=str(ROOT/'triton'/name))
                for key in ['FOREST_NATIVE_GATED_NORM','PRM_CUDA_MEMORY_FRACTION','PYTHONPATH']:env.pop(key,None)
                command=[info['python'],'-m','experiments.group2_nomemory_20261115.worker','--job',name,'--phase',phase,
                    '--base-job',job['base'],'--shard',str(job['shard']),'--shards',str(job['shards'])]
                log=folder/f'worker-{time.time_ns()}.log'
                with log.open('w') as stream:
                    child=subprocess.Popen(command,cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                save(folder/f'launch-{child.pid}.json',dict(pid=child.pid,gpu=gpu,command=command,started=time.time()))
                active[gpu]=dict(process=child,name=name,log=log)
            state.update(phase=phase,updated=time.time(),pending=pending,
                active={str(g):dict(pid=j['process'].pid,job=j['name'],log=str(j['log'])) for g,j in active.items()},failed=failures)
            save(ROOT/'status.json',state)
            health=host_health()
            if health['under_oom'] or health['oom_kill']!=baseline['oom_kill'] or health['resident']>.8*health['limit']:
                save(ROOT/'resource-stop.json',dict(reason='Host memory pressure',health=health))
                failures.append(dict(reason='Host memory protection'));pending=[]
            if failures and not active:break
            if pending or active:time.sleep(5)
        assert not failures,failures
        for name in jobs:
            proof=read(ROOT/phase/name/'verification.json');assert proof['status'] in ('passed','completed')
        save(ROOT/f'{phase}-gate.json',dict(status='passed',jobs=len(jobs),finished=time.time(),inputs_sha256=sha(ROOT/'inputs.json')))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLBACKEND='Agg')
    result=subprocess.run([PYTHON,'-m','experiments.group2_nomemory_20261115.report'],cwd=ROOT/'code',env=env,
        stdout=(ROOT/'report.log').open('w'),stderr=subprocess.STDOUT)
    assert result.returncode==0,'Final report failed; raw predictions preserved'
    state.update(status='completed',phase='finished',active={},finished=time.time(),report=str(OUT/'metrics.json'))
    save(ROOT/'status.json',state)

if __name__=='__main__':
    try:main()
    except BaseException:
        p=ROOT/'status.json';state=read(p) if p.exists() else {}
        state.update(status='failed',error=traceback.format_exc(),updated=time.time());save(p,state)
        raise
