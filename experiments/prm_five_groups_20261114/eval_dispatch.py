"""Persistent official-baseline dispatch; never report unimplemented groups done.

Adopts the existing Robometer job statuses, then runs the five other official
models one extra worker per safe GPU. Future experiment conditions are recorded
as pending implementation, rather than silently omitted from completion checks.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from .prepare_cache import BASE,save
from .eval_common import EVAL,aggregate
from .overlap_guard import resources,pause_reasons,alive

CODE=Path(__file__).parents[2]
ORDER=['vlac-2b','robobrain25-4b','dopamine-3b','procvlm-2b','r2vlm-alfred-rl']
DONE={'completed','completed_with_input_blockers'}
PENDING_IMPLEMENTATION=['FoREST group1 final-checkpoint adapter',
    'group2 memory removal and shared seeded disturbance manifests/runners',
    'group3 Recover inputs and event metrics integration',
    'group4 O12/S13 instruction-matched counterfactual truth and runner',
    'group5 five-FPS-stream runners and common-time scoring']


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def spec(model,shard):
    python='/cache/robot-baselines/envs/'+('vlac' if model=='vlac-2b' else 'procvlm')+'/bin/python'
    return {'name':f'{model}-group1-{shard}','model':model,'shard':shard,'group':1,
        'python':python,'module':'experiments.prm_five_groups_20261114.official_baseline',
        'memory_fraction':.30,'batch':1}


def canary_ready(job,states):
    # New execution allocation must finish a real episode in shard 0 before
    # replicating that model on the other GPUs. Formal predictions are retained.
    if job['shard']==0:return True
    first=states.get(f"{job['model']}-group1-0",{})
    return first.get('status') in DONE|{'running','paused_by_resource_guard'} and first.get('completed_episodes',0)>0


def descends_from(pid,parent):
    seen=set()
    while pid not in seen and pid>1:
        if pid==parent:return True
        seen.add(pid)
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[1])
        except FileNotFoundError:return False
    return False


def gpu_processes():
    uuids={r.split(',')[1].strip():int(r.split(',')[0]) for r in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()}
    result={i:[] for i in range(4)}
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,timeout=10).splitlines():
        uuid,pid=line.split(',');result[uuids[uuid.strip()]].append(int(pid))
    return result


def permitted_phase(training,gpu,now):
    # The training controller waits synchronously after its verified handoff,
    # so its status timestamp intentionally stops advancing at that point.
    if len(training.get('training_completed',{}))==4:return True
    if now-training.get('updated',0)>90:return False
    return any(gpu in group['gpus'] and group['phase']=='training' and alive(group.get('pid'))
               for group in training.get('groups',[]))


def eligible_gpu(training,gpu,sample,processes,robometer,initial_oom,now):
    if not permitted_phase(training,gpu,now) or pause_reasons(sample,gpu,initial_oom):return False
    # Do not race the existing guard's automatic resume of a Robometer shard.
    if robometer.get('status') not in DONE or alive(robometer.get('pid')):return False
    group=next((g for g in training.get('groups',[]) if gpu in g['gpus']),None)
    if group and group['phase']=='training':
        return all(descends_from(pid,group['pid']) for pid in processes[gpu])
    return not processes[gpu]


def launch(job,gpu):
    path=EVAL/'jobs'/f"{job['name']}.json"
    previous=read(path,{})
    if alive(previous.get('pid')):raise RuntimeError('Previous worker is still alive')
    if previous and previous.get('status')!='paused_by_resource_guard':
        raise RuntimeError('Only a verified resource pause can restart automatically')
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HOME='/cache/cvpzhr/cache/huggingface',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONNOUSERSITE='1',
        PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    environment.pop('PYTHONPATH',None)
    with path.with_suffix('.log').open('a') as log:
        proc=subprocess.Popen([job['python'],'-m',job['module'],job['model'],'--shard',str(job['shard']),
            '--memory-fraction',str(job['memory_fraction'])],cwd=CODE,env=environment,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(job,gpu=gpu,pid=proc.pid,started=time.time())
    save(EVAL/'dispatch-launches'/f"{job['name']}-{proc.pid}.json",receipt)
    return proc,receipt


def wait_existing():
    while True:
        state=read(EVAL/'dispatcher-status.json',{})
        if state.get('all_experiments_verified'):return
        if not alive(state.get('pid')):raise RuntimeError('Existing dispatcher exited without full verification; inspect status')
        time.sleep(10)


def run():
    EVAL.mkdir(parents=True,exist_ok=True);(EVAL/'jobs').mkdir(exist_ok=True)
    lock=(EVAL/'dispatcher.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return wait_existing()
    prior=read(EVAL/'dispatcher-status.json',{})
    # On recovery do not orphan/adopt unverified old GPU subprocess trees.
    for p in (EVAL/'dispatch-launches').glob('*.json'):
        receipt=read(p)
        if alive(receipt['pid']):raise RuntimeError(f"Existing worker {receipt['pid']} requires inspection before dispatcher recovery")
    preflight=read(EVAL/'official-baseline-preflight.json',{})
    assert preflight.get('status')=='passed' and set(preflight['models'])==set(ORDER), 'Official model preflight missing'
    (EVAL/'dispatcher.pid').write_text(str(os.getpid()))
    jobs=[spec(model,shard) for model in ORDER for shard in range(4)]
    save(EVAL/'dispatch-plan.json',{'official_baseline_jobs':jobs,'adopted_robometer_shards':4,
        'pending_implementation':PENDING_IMPLEMENTATION,'all_five_groups_ready':False})
    initial_oom=resources()['oom_kill'];active={};safe_since={i:None for i in range(4)}
    launches={j['name']:[] for j in jobs};last_launch=0;started=time.time();aggregated=set()
    while True:
        now=time.time();training=read(BASE/'status.json',{});sample=resources();processes=gpu_processes()
        states={j['name']:read(EVAL/'jobs'/f"{j['name']}.json",{}) for j in jobs}
        for gpu,(job,proc,receipt) in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                state=read(EVAL/'jobs'/f"{job['name']}.json",{})
                if state.get('status') not in DONE|{'paused_by_resource_guard','failed'}:
                    save(EVAL/'jobs'/f"{job['name']}.json",dict(state,status='failed',pid=proc.pid,
                        error=f'Worker exited {rc} without terminal status; inspect log',updated=now))
                del active[gpu];safe_since[gpu]=None;continue
            reasons=pause_reasons(sample,gpu,initial_oom)
            if not permitted_phase(training,gpu,now):reasons.append('yield_to_training_queue_transition')
            if (EVAL/'pause-all-official-baseline').exists():reasons.append('manual_pause_all')
            if reasons:
                (EVAL/f"pause-{job['name']}").touch()
                save(EVAL/'resource-pauses'/f"{job['name']}.json",{'reasons':reasons,'at':now})
        for gpu in range(4):
            if gpu in active:continue
            robo=read(EVAL/'jobs'/f'robometer-4b-group1-{gpu}.json',{})
            safe=eligible_gpu(training,gpu,sample,processes,robo,initial_oom,now)
            if (EVAL/'pause-all-official-baseline').exists():safe=False
            if not safe:safe_since[gpu]=None;continue
            if safe_since[gpu] is None:safe_since[gpu]=now
            if now-safe_since[gpu]<60 or now-last_launch<60:continue
            busy={job['name'] for job,_,_ in active.values()}
            candidate=None
            for job in jobs:
                old=states[job['name']]
                if job['name'] in busy or old.get('status') in DONE|{'failed'} or alive(old.get('pid')):continue
                if old and old.get('status')!='paused_by_resource_guard':continue
                if not canary_ready(job,states):continue
                launches[job['name']]=[t for t in launches[job['name']] if now-t<3600]
                if len(launches[job['name']])>=3:continue
                candidate=job;break
            if candidate:
                (EVAL/f"pause-{candidate['name']}").unlink(missing_ok=True)
                proc,receipt=launch(candidate,gpu);active[gpu]=(candidate,proc,receipt)
                launches[candidate['name']].append(now);last_launch=now
        for model in ORDER:
            if model not in aggregated and all(states[f'{model}-group1-{s}'].get('status') in DONE for s in range(4)):
                aggregate(model);aggregated.add(model)
        save(EVAL/'dispatcher-status.json',{'status':'running' if active else 'waiting_for_safe_slot_or_implementation',
            'pid':os.getpid(),'started':started,'updated':now,'all_experiments_verified':False,
            'active':{g:r for g,(_,_,r) in active.items()},'baseline_models_aggregated':sorted(aggregated),
            'baseline_job_status':{j['name']:states[j['name']].get('status','pending') for j in jobs},
            'pending_implementation':PENDING_IMPLEMENTATION,'resource_sample':sample})
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['run']);parser.parse_args()
    try:run()
    except BaseException:
        save(EVAL/'dispatcher-error.json',{'error':traceback.format_exc(),'at':time.time()})
        # Stop only this dispatcher's supplemental jobs cooperatively.
        for model in ORDER:
            for shard in range(4):(EVAL/f'pause-{model}-group1-{shard}').touch()
        raise
