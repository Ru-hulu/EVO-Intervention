"""Persistent five-group dispatch with first-group priority and audited dependencies."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

from .prepare_cache import BASE,save,sha
from .eval_common import EVAL,aggregate
from .overlap_guard import resources,pause_reasons,alive
from .forest_adapter import check_final_proof
from .gpu_ownership import GroupOwnership
from .evaluation_priority import reserved, ReservedEvaluationOwnership
from .followup_schedule import (planned_followups, followup_ready, official_jobs_terminal,
                               finalize_official_handoff, release_count_reservation)
from .eval_handoff import adopt as adopt_evaluations

CODE=Path(__file__).parents[2]
ORDER=['vlac-2b','robobrain25-4b','dopamine-3b','procvlm-2b','r2vlm-alfred-rl']
DONE={'completed','completed_with_input_blockers'}
PENDING_IMPLEMENTATION=[]


def read(path,default=None):return json.loads(path.read_text()) if path.exists() else default


def spec(model,shard):
    python='/cache/robot-baselines/envs/'+('vlac' if model=='vlac-2b' else 'procvlm')+'/bin/python'
    return {'name':f'{model}-group1-{shard}','model':model,'shard':shard,'group':1,
        'python':python,'module':'experiments.prm_five_groups_20261114.official_baseline',
        'memory_fraction':.30,'batch':1,'args':[model,'--shard',str(shard)],
        'canary_name':f'{model}-group1-0'}


def forest_spec(suite,shard):
    return {'name':f'forest-{suite.lower()}-group1-{shard}','model':'forest','suite':suite,
        'shard':shard,'group':1,'python':'/cache/cvpzhr/envs/forest-labels/bin/python',
        'module':'experiments.prm_five_groups_20261114.forest_eval','memory_fraction':.30,'batch':1,
        'args':['--suite',suite,'--shard',str(shard)],'canary_name':f'forest-{suite.lower()}-group1-0'}


def planned_jobs():
    # Finished suite weights may proceed when a current shard releases a slot;
    # no running model is preempted. Unfinished suites remain ineligible.
    return [forest_spec(s,sid) for s in ['State','Order','Count'] for sid in range(4)]+[
        spec(model,sid) for model in ORDER for sid in range(4)]+planned_followups()


def checkpoint_ready(job,training):
    if job['model']!='forest':return True
    suite=job['suite'];record=training.get('training_completed',{}).get(suite)
    if not record:return False
    path=BASE/suite/'checkpoint-verification.json'
    if not path.exists():return False
    proof=read(path);expected=Path(f'/cache/cvpzhr/outputs/forest-{suite.lower()}-qwen35-clip-depth-e1-20261114/epoch-01.pt')
    check_final_proof(proof,suite,expected)
    assert record['sha256']==proof['sha256'] and record['checkpoint']==proof['checkpoint']
    # The worker performs full file/tensor hashing before any prediction.
    return expected.is_file()


def canary_ready(job,states):
    # New execution allocation must finish a real episode in shard 0 before
    # replicating that model on the other GPUs. Formal predictions are retained.
    if job['shard']==0:return True
    first=states.get(job['canary_name'],{})
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


def gpu_minors():
    uuids={r.split(',')[1].strip():int(r.split(',')[0]) for r in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True,timeout=10).splitlines()}
    report=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=10))
    return {uuids[g.findtext('uuid')]:int(g.findtext('minor_number')) for g in report.findall('gpu')}


def permitted_phase(training,gpu,now):
    # The training controller waits synchronously after its verified handoff,
    # so its status timestamp intentionally stops advancing at that point.
    if len(training.get('training_completed',{}))==4:return True
    if now-training.get('updated',0)>90:return False
    if reserved(BASE,gpu):return True
    from .state_acceleration import permits_existing_evaluation
    # The maintenance receipt bridges the few seconds between torchrun exit and
    # the training controller publishing its next status; keep evals running.
    if permits_existing_evaluation(BASE,gpu):return True
    return any(gpu in group['gpus'] and group['phase']=='training' and alive(group.get('pid'))
               for group in training.get('groups',[]))


def eligible_gpu(training,gpu,sample,processes,robometer,initial_oom,now,ownership=None,reservation_ownership=None,minors=None):
    if not permitted_phase(training,gpu,now) or pause_reasons(sample,gpu,initial_oom):return False
    from .state_acceleration import permits_existing_evaluation
    if permits_existing_evaluation(BASE,gpu):
        return False  # Existing evaluations continue; do not race a validation process.
    if reserved(BASE,gpu):
        group=next((g for g in training.get('groups',[]) if gpu in g['gpus']),{})
        if group.get('phase')!='evaluation_reserved':return False
        return reservation_ownership is not None and reservation_ownership.permits(BASE,gpu,robometer,processes[gpu],minors[gpu],now)
    # Do not race the existing guard's automatic resume of a Robometer shard.
    if robometer.get('status') not in DONE or alive(robometer.get('pid')):return False
    group=next((g for g in training.get('groups',[]) if gpu in g['gpus']),None)
    if group and group['phase']=='training':
        if ownership is not None:return ownership.permits(group,gpu,processes)
        return bool(processes[gpu]) and all(descends_from(pid,group['pid']) for pid in processes[gpu])
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
        proc=subprocess.Popen([job['python'],'-m',job['module'],*job['args'],
            '--memory-fraction',str(job['memory_fraction'])],cwd=CODE,env=environment,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(job,gpu=gpu,pid=proc.pid,started=time.time(),dispatcher_pid=os.getpid(),
                 resource_admission_passed=True,first_group_priority_cleared=job['group']>1)
    save(EVAL/'dispatch-launches'/f"{job['name']}-{proc.pid}.json",receipt)
    return proc,receipt


def wait_existing():
    while True:
        state=read(EVAL/'dispatcher-status.json',{})
        if state.get('all_experiments_verified'):return
        if not alive(state.get('pid')):raise RuntimeError('Existing dispatcher exited without full verification; inspect status')
        time.sleep(10)


def run(adopt=None):
    EVAL.mkdir(parents=True,exist_ok=True);(EVAL/'jobs').mkdir(exist_ok=True)
    lock=(EVAL/'dispatcher.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return wait_existing()
    prior=read(EVAL/'dispatcher-status.json',{})
    # On recovery do not orphan/adopt unverified old GPU subprocess trees.
    if not adopt:
        for p in (EVAL/'dispatch-launches').glob('*.json'):
            receipt=read(p)
            if alive(receipt['pid']):raise RuntimeError(f"Existing worker {receipt['pid']} requires inspection before dispatcher recovery")
    preflight=read(EVAL/'official-baseline-preflight.json',{})
    assert preflight.get('status')=='passed' and set(preflight['models'])==set(ORDER), 'Official model preflight missing'
    (EVAL/'dispatcher.pid').write_text(str(os.getpid()))
    jobs=planned_jobs();boot_sha=sha(__file__)
    save(EVAL/'dispatch-plan.json',{'baseline_jobs':[j for j in jobs if j['group']==1],
        'followup_jobs':[j for j in jobs if j['group']>1], 'followup_job_count':440,'official_baseline_job_count':20,
        'final_suite_forest_job_count':12,'adopted_robometer_shards':4,
        'pending_implementation':PENDING_IMPLEMENTATION,'all_five_groups_code_implemented':True,
        'all_five_groups_gpu_validated':False})
    initial_oom=resources()['oom_kill'];active={};safe_since={i:None for i in range(4)}
    if adopt:
        active,handoff=adopt_evaluations(adopt,EVAL/'jobs',jobs)
        initial_oom=handoff['initial_oom']
        save(EVAL/'dispatcher-adoption.json',{'status':'passed','pid':os.getpid(),
             'adopted_workers':{g:r for g,(_,_,r) in active.items()},'handoff':str(adopt),'at':time.time()})
    minors=gpu_minors()
    ownership=GroupOwnership()
    reservation_ownership=ReservedEvaluationOwnership()
    launches={j['name']:[] for j in jobs};last_launch=0;started=time.time();aggregated=set()
    cpu=None;cpu_action=None;prepared_at=0;reported={};priority_cleared=False;dependency_blockers={}
    completion={'all_experiments_verified':False};last_completion=0
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def cpu_launch(module,args,action):
        with (EVAL/'followup-cpu.log').open('a') as log:
            proc=subprocess.Popen(['/cache/cvpzhr/envs/forest-labels/bin/python','-m',module,*args],
                cwd=CODE,env=cpu_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        return proc,action
    while True:
        now=time.time();training=read(BASE/'status.json',{});sample=resources();processes=gpu_processes()
        ownership.refresh(training,processes,now,BASE,minors)
        states={j['name']:read(EVAL/'jobs'/f"{j['name']}.json",{}) for j in jobs};readiness_errors={}
        for shard in range(4):states[f'robometer-4b-group1-{shard}']=read(EVAL/'jobs'/f'robometer-4b-group1-{shard}.json',{})
        if cpu is not None and cpu.poll() is not None:
            if cpu.returncode:
                save(EVAL/'followup-cpu-errors'/f'{time.time_ns()}.json',{'action':cpu_action,'exit':cpu.returncode,'at':now})
                if cpu_action and cpu_action[0]=='report':reported[cpu_action[1]]='failed_requires_inspection'
            cpu=None;cpu_action=None
        if cpu is None and now-prepared_at>300:
            cpu,cpu_action=cpu_launch('experiments.prm_five_groups_20261114.experiment_inputs',[],('prepare',))
            prepared_at=now
        readiness=read(EVAL/'conditions/readiness.json',{})
        handoff_path=EVAL/'group1-official-handoff.json'
        if not priority_cleared and official_jobs_terminal(states) and not any(j['group']==1 and j['model']!='forest' for j,_,_ in active.values()):
            if not handoff_path.exists():finalize_official_handoff(states)
            handoff=read(handoff_path);assert handoff['status']=='passed' and sha(handoff['report_path'])==handoff['report_sha256']
            priority_cleared=release_count_reservation(training)
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
        reload_requested=(EVAL/'dispatcher-reload-when-idle.json').exists()
        for gpu in range(4):
            if gpu in active:continue
            if reload_requested:continue
            robo=read(EVAL/'jobs'/f'robometer-4b-group1-{gpu}.json',{})
            safe=eligible_gpu(training,gpu,sample,processes,robo,initial_oom,now,ownership,reservation_ownership,minors)
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
                if job['group']>1:
                    try:ready,reason=followup_ready(job,readiness,priority_cleared)
                    except (AssertionError,KeyError,ValueError) as exc:
                        readiness_errors[job['name']]=str(exc);continue
                    if not ready:
                        dependency_blockers[job['name']]=reason;continue
                    dependency_blockers.pop(job['name'],None)
                try:
                    ready=checkpoint_ready(job,training)
                except (AssertionError,KeyError,ValueError) as exc:
                    readiness_errors[job['name']]=f'{type(exc).__name__}: {exc}';continue
                if not ready:continue
                if job['group']==1 and not canary_ready(job,states):continue
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
        if 'forest' not in aggregated and all(states[j['name']].get('status') in DONE for j in jobs if j['model']=='forest' and j['group']==1):
            aggregate('forest');aggregated.add('forest')
            from .group1_report import build as refresh_baseline_report
            refresh_baseline_report()
        if cpu is None:
            for job in jobs:
                if job['group']==1:continue
                key=(job['group'],job['condition'],job['model'])
                related=[j for j in jobs if (j['group'],j.get('condition'),j['model'])==key]
                finished=tuple((j['name'],states[j['name']].get('finished')) for j in related if states[j['name']].get('status') in DONE)
                revision=readiness.get('conditions',{}).get(f'group{key[0]}/{key[1]}',{}).get('event_annotation_revision')
                signature=(finished,revision)
                if not finished or reported.get(key)==signature or reported.get(key)=='failed_requires_inspection':continue
                reported[key]=signature
                cpu,cpu_action=cpu_launch('experiments.prm_five_groups_20261114.experiment_report',
                    ['--group',str(key[0]),'--condition',key[1],'--model',key[2]],('report',key))
                break
        if not active and cpu is None and now-last_completion>60:
            from .experiment_completion import build as completion_audit
            completion=completion_audit(states);last_completion=now
        save(EVAL/'dispatcher-status.json',{'status':'running' if active else 'waiting_for_safe_slot_or_dependencies',
            'pid':os.getpid(),'started':started,'updated':now,'all_experiments_verified':completion['all_experiments_verified'],
            'boot_code_sha256':boot_sha,'reload_requested':reload_requested,
            'readiness_errors':readiness_errors,
            'gpu_ownership':ownership.status,
            'reserved_evaluation_ownership':reservation_ownership.status,
            'active':{g:r for g,(_,_,r) in active.items()},'baseline_models_aggregated':sorted(aggregated),
            'baseline_job_status':{j['name']:states[j['name']].get('status','pending') for j in jobs if j['group']==1},
            'followup_job_status':{j['name']:states[j['name']].get('status','pending') for j in jobs if j['group']>1},
            'first_group_official_priority_cleared':priority_cleared,'dependency_blockers':dependency_blockers,
            'input_readiness':readiness,'cpu_action':cpu_action,
            'pending_implementation':PENDING_IMPLEMENTATION,'resource_sample':sample})
        if completion['all_experiments_verified']:
            return
        if reload_requested and not active and cpu is None:
            # Future source upgrades replace only this idle CPU scheduler. No
            # worker receives a signal; original training and evaluation stay live.
            if not any(alive(read(p)['pid']) for p in (EVAL/'dispatch-launches').glob('*.json')):
                request=read(EVAL/'dispatcher-reload-when-idle.json')
                save(EVAL/'dispatcher-reload-history'/f'{time.time_ns()}.json',{
                    'pid':os.getpid(),'old_boot_code_sha256':boot_sha,'new_code_sha256':sha(__file__),
                    'request':request,'active_workers':0,'at':time.time()})
                (EVAL/'dispatcher-reload-when-idle.json').unlink()
                os.execv(sys.executable,[sys.executable,'-m','experiments.prm_five_groups_20261114.eval_dispatch','run'])
        time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['run']);parser.add_argument('--adopt');args=parser.parse_args()
    try:run(args.adopt)
    except BaseException:
        save(EVAL/'dispatcher-error.json',{'error':traceback.format_exc(),'at':time.time()})
        # Stop only this dispatcher's supplemental jobs cooperatively.
        for job in planned_jobs():(EVAL/f"pause-{job['name']}").touch()
        raise
