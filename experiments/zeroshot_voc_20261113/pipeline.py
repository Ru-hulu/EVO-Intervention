"""Four GPU queue; isolated model shards, no interference with external GPU jobs."""
import fcntl
import subprocess
import time
from common import *

ORDER=['r2vlm-alfred-rl','procvlm-2b','robometer-4b','robobrain25-8b','dopamine-8b','robobrain25-4b','dopamine-3b','vlac-2b','vlac-8b']


def gpu_free(gpu):
    r=subprocess.run(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    mem,util=map(int,r.stdout.strip().split(','))
    return mem<1000 and util<10


def report():
    results={}
    for model in MODELS:
        if all((ROOT/'jobs'/f'{model}-{s}.json').exists() and read(ROOT/'jobs'/f'{model}-{s}.json')['status']=='completed' for s in range(4)):
            results[model]=aggregate(model)
    lines=['# Zero-shot clean validation: Pearson / Spearman','',
           'Official checkpoints. Per-episode correlations with original PT, then equal episode mean. No smoothing.',
           'GRM models use the true final main-view reference; other methods use only observed information.','',
           '| Model | State P / S | Order P / S | Count P / S | Overall P / S | Valid / total episodes | Invalid frames |',
           '|---|---|---|---|---|---|---|']
    def fmt(r):return 'undefined' if r['pearson'] is None else f"{r['pearson']:.4f} / {r['spearman']:.4f}"
    for name in MODELS:
        if name not in results:continue
        r=results[name];o=r['overall']
        lines.append('| '+' | '.join([name]+[fmt(r['by_suite'][s]) for s in ['State','Order','Count']]+[fmt(o),f"{o['valid_episodes']} / {o['episodes']}",str(r['invalid_predictions'])])+' |')
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'comparison.json',results)
    return list(results)


def main():
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    read(ROOT/'preflight.json');live={};failed=set();completed=set();logs=[]
    jobs=[(m,s) for s in range(4) for m in ORDER]
    for m,s in jobs:
        p=ROOT/'jobs'/f'{m}-{s}.json'
        if p.exists():
            state=read(p)
            if state['status']=='completed':completed.add((m,s))
            elif state['status']=='failed':failed.add(m)
            else:
                try:os.kill(state['pid'],0)
                except ProcessLookupError: pass
                else:raise RuntimeError('Old worker still alive; refuse duplicate launch: '+str(p))
    last_report=0;started=now()
    while True:
        for gpu,(job,proc) in list(live.items()):
            rc=proc.poll()
            if rc is None:continue
            m,s=job;state=read(ROOT/'jobs'/f'{m}-{s}.json') if (ROOT/'jobs'/f'{m}-{s}.json').exists() else {}
            if rc==0 and state.get('status')=='completed':completed.add(job)
            else:failed.add(m)
            del live[gpu]
        pending=[j for j in jobs if j not in completed and j[0] not in failed and j not in [x[0] for x in live.values()]]
        for gpu in range(4):
            if gpu in live or not pending or not gpu_free(gpu):continue
            # Validate shard0 before allowing other shards of a model.
            eligible=[j for j in pending if j[1]==0 or (ROOT/'jobs'/f'{j[0]}-0-audit.json').exists()]
            if not eligible:continue
            m,s=eligible[0];pending.remove((m,s));cfg=MODELS[m]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',
                     OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     PYTHONUNBUFFERED='1',HF_HOME='/cache/cvpzhr/cache/huggingface',
                     VLLM_WORKER_MULTIPROC_METHOD='spawn',EVQA_DATASET_ROOT='/cache/robot-baselines/data/procvlm')
            log=(ROOT/'logs'/f'{m}-{s}.log');log.parent.mkdir(exist_ok=True);handle=log.open('a');logs.append(handle)
            proc=subprocess.Popen([cfg['python'],'-u',str(ROOT/'worker.py'),m,'--shard',str(s)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            live[gpu]=((m,s),proc)
        if time.time()-last_report>30:
            finished=report();last_report=time.time()
        else:finished=[m for m in MODELS if all((m,s) in completed for s in range(4))]
        state=dict(status='running' if live or pending else 'failed' if failed else 'completed',started_at=started,updated_at=now(),
                   completed_jobs=len(completed),total_jobs=len(jobs),completed_models=finished,failed_models=sorted(failed),
                   active={str(g):dict(model=j[0],shard=j[1],pid=p.pid) for g,(j,p) in live.items()},
                   result_dir=str(OUT),protocol_sha256=sha(ROOT/'protocol.json'))
        write(ROOT/'status.json',state)
        if not live and not pending:report();break
        time.sleep(5)


if __name__=='__main__': main()
