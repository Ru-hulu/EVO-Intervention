"""Two GPU pairs, one model/suite per pair, no interruption of existing FoREST."""
import fcntl
import os
import subprocess
import time
import traceback
from .common import *
from .worker import contexts
from .report import report
from experiments.soc_disrupted_eval_20261115.guard import process_identity,alive,gpu_processes,host_sample
from experiments.recover_validation_20261114.common import safe


def forest_released(status,workers):
    return status.get('status')=='completed' and not alive(status.get('controller')) and all(not alive(w.get('process')) for w in workers)


def run():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'status.json').exists():assert not alive(read(ROOT/'status.json').get('controller'))
    for p in ROOT.glob('jobs/*/*/status.json'):assert not alive(read(p).get('process'))
    inputs=read(ROOT/'inputs.json');ih=sha(ROOT/'inputs.json');assert read(ROOT/'preparation.json')['inputs_sha256']==ih
    for p,h in read(ROOT/'source-checksums.json')['files'].items():assert sha(ROOT/'code'/p)==h,p
    tasks=inputs['tasks'];controller=process_identity(os.getpid());(ROOT/'controller.pid').write_text(str(os.getpid())+'\n')
    snapshot=report();complete={k for k,v in snapshot['tasks'].items() if v['complete'] and k in tasks}
    pending=[k for k in inputs['task_order'] if k not in complete];active={};failures=[];pairs={};stable={};last_launch={}
    initial=host_sample();state=dict(status='starting',controller=controller,started=time.time(),inputs_sha256=ih)
    def update(**kw):state.update(kw,updated=time.time());save(ROOT/'status.json',state)
    update()
    forest_summary=read(FOREST/'status.json');last_report=0
    while True:
        now=time.time()
        for name,item in list(active.items()):
            code=item['child'].poll()
            if code is None:continue
            folder=ROOT/'jobs'/item['task']/f"shard{item['shard']}"
            status=read(folder/'status.json') if (folder/'status.json').exists() else {}
            pair=pairs[item['pair']]
            if code==0 and status.get('status')=='completed':pair['completed'].append(item['shard'])
            elif code==75:
                pair['pending'].append(item['shard']);pair['cooldown']=now+60
            else:
                issue=dict(task=item['task'],shard=item['shard'],returncode=code,log=item['log'])
                failures.append(issue);pair['failed']=True
            del active[name]
        for number,pair in list(pairs.items()):
            here=[v for v in active.values() if v['pair']==number]
            if not here and (pair.get('failed') or (not pair['pending'])):
                if not pair.get('failed'):
                    report(pair['task']);complete.add(pair['task'])
                    save(ROOT/'task-completions'/f"{pair['task']}.json",dict(status='completed',finished=now,
                        shards=sorted(pair['completed']),gpus=pair['gpus'],inputs_sha256=ih))
                del pairs[number]
        sample=host_sample();processes=gpu_processes()
        forest_summary=read(FOREST/'status.json')
        forest_workers=[read(p) for p in (FOREST/'jobs').glob('*/status.json')]
        released=forest_released(forest_summary,forest_workers)
        allowed={0:(2,3)}
        if released:allowed[1]=(0,1)
        for number,gpus in allowed.items():
            if number in pairs or not pending or (ROOT/'pause-all').exists():continue
            idle=all(not processes[g] and safe(sample,g,initial['oom_kill']) for g in gpus)
            if idle:stable[number]=stable.get(number,now)
            else:stable.pop(number,None)
            if idle and now-stable[number]>=60:
                task=pending.pop(0);job=tasks[task]
                pairs[number]=dict(task=task,gpus=list(gpus),pending=list(range(len(job['shards']))),completed=[],started=now)
                stable.pop(number,None)
        for number,pair in pairs.items():
            if pair.get('failed') or pair.get('cooldown',0)>now or (ROOT/'pause-all').exists():continue
            task=pair['task'];job=tasks[task]
            for side,gpu in enumerate(pair['gpus']):
                if now-last_launch.get(gpu,0)<15 or not safe(sample,gpu,initial['oom_kill']):continue
                here=[v for v in active.values() if v['gpu']==gpu]
                if len(here)>=job['replicas_per_gpu']:continue
                if not set(processes[gpu])<=contexts(gpu,controller):continue
                if any(not (ROOT/'jobs'/v['task']/f"shard{v['shard']}"/'context.json').exists() for v in here):continue
                candidates=[k for k in pair['pending'] if k%2==side]
                if not candidates:continue
                shard=min(candidates)
                # First GPU replica must pass native numerical validation and
                # finish a real trajectory before the model is replicated here.
                if shard>=2:
                    proof=ROOT/'jobs'/task/f'shard{side}'/'first-real-episode.json'
                    if not proof.exists() or read(proof)['status']!='passed':continue
                folder=ROOT/'jobs'/task/f'shard{shard}';folder.mkdir(parents=True,exist_ok=True)
                ticket=folder/f'ticket-{time.time_ns()}.json'
                save(ticket,dict(task=task,shard=shard,gpu=gpu,controller=controller,initial_oom=initial['oom_kill'],
                    inputs_sha256=ih,allocator_fraction=.20 if job['model']=='procvlm-2b' else .35))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
                for k in ('PYTHONPATH','FOREST_NATIVE_GATED_NORM','PRM_CUDA_MEMORY_FRACTION','PYTORCH_CUDA_ALLOC_CONF'):env.pop(k,None)
                log=folder/f'worker-{time.time_ns()}.log';python=inputs['registry'][task]['python']
                with log.open('w') as f:
                    child=subprocess.Popen([python,'-m',PACKAGE+'.worker','--ticket',str(ticket)],cwd=ROOT/'code',env=env,
                        stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                name=f'{task}/shard{shard}';active[name]=dict(child=child,gpu=gpu,pair=number,task=task,shard=shard,log=str(log))
                pair['pending'].remove(shard);last_launch[gpu]=now
        update(status='running' if active or pairs else 'waiting_for_safe_pair_or_forest',
            completed_tasks=sorted(complete),pending_tasks=pending,pairs=pairs,failures=failures,
            active={k:{a:v[a] for a in ('gpu','pair','task','shard','log')}|dict(pid=v['child'].pid) for k,v in active.items()},
            external_forest=dict(status=forest_summary['status'],suite=forest_summary.get('suite'),gpus=[0,1],released=released))
        if now-last_report>60:report();last_report=now
        if (ROOT/'pause-all').exists() and not active:update(status='paused');return
        if not pending and not active and not pairs:
            if forest_summary['status']=='completed' and released:
                result=report();update(status='completed' if result['status']=='completed' and not failures else 'completed_with_failures',
                    report=str(OUT/'report.json'),finished=time.time());return
            if forest_summary['status'] in ('failed','paused'):
                update(status='blocked_on_external_forest',report=str(OUT/'report.json'));return
        time.sleep(5)


if __name__=='__main__':
    try:run()
    except BaseException:
        status=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        status.update(status='failed',error=traceback.format_exc(),updated=time.time());save(ROOT/'status.json',status);raise
