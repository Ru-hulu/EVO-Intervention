"""Bounded official baseline evaluation alongside healthy two-GPU training.

Only the extra evaluation workers cooperate with pause flags. This guard never
signals training, changes its configuration, or retries an unexplained failure.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from .prepare_cache import BASE, save
from .eval_common import EVAL, aggregate

CODE=Path(__file__).parents[2]
PYTHON='/cache/cvpzhr/envs/robometer/bin/python'
DONE={'completed','completed_with_input_blockers'}


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def alive(pid):
    if not pid:return False
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def resources():
    lines=subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10).splitlines()
    gpus={}
    for line in lines:
        i,used,total,util,temp=map(int,line.split(','))
        gpus[i]={'used_mib':used,'total_mib':total,'utilization':util,'temperature':temp}
    root=Path('/sys/fs/cgroup/memory')
    stat={k:int(v) for k,v in (line.split() for line in (root/'memory.stat').read_text().splitlines())}
    oom={k:int(v) for k,v in (line.split() for line in (root/'memory.oom_control').read_text().splitlines())}
    usage=int((root/'memory.usage_in_bytes').read_text())
    limit=int((root/'memory.limit_in_bytes').read_text())
    return {'gpus':gpus,'memory_limit':limit,'memory_usage':usage,
        'memory_working_set':max(0,usage-stat['total_inactive_file']),
        'resident_and_shmem':stat['total_rss']+stat.get('total_shmem',0),
        'oom_kill':oom['oom_kill'],'under_oom':oom['under_oom'],
        'disk_free':shutil.disk_usage('/cache').free}


def pause_reasons(sample,gpu,initial_oom):
    reasons=[];v=sample['gpus'][gpu]
    if v['total_mib']-v['used_mib']<16384:reasons.append('less_than_16_GiB_GPU_headroom')
    if v['temperature']>=80:reasons.append('GPU_temperature_at_least_80C')
    if sample['resident_and_shmem']>.65*sample['memory_limit']:reasons.append('container_resident_memory_high')
    if sample['memory_working_set']>.85*sample['memory_limit']:reasons.append('container_working_set_high')
    if sample['under_oom'] or sample['oom_kill']>initial_oom:reasons.append('container_OOM_requires_inspection')
    if sample['disk_free']<100*2**30:reasons.append('less_than_100_GiB_disk_free')
    return reasons


def launch(shard):
    """The common launch lock plus the worker's own flock prevents duplicates."""
    with (EVAL/'early-launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=EVAL/'early-launch.json';history=read(path,{'launches':[]})
        for item in history['launches']:
            if item['shard']==shard and alive(item['pid']):return item['pid']
        job=EVAL/'jobs'/f'robometer-4b-group1-{shard}.json'
        prior=read(job,{})
        if alive(prior.get('pid')):return prior['pid']
        if prior and prior.get('status')!='paused_by_resource_guard':return None
        if prior:
            archive=EVAL/'jobs/history'/str(time.time_ns());archive.mkdir(parents=True)
            for p in [job,job.with_name(job.stem+'-initialization.json')]:
                if p.exists():shutil.copy2(p,archive/p.name)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
        command=[PYTHON,'-m','experiments.prm_five_groups_20261114.robometer_eval',
            '--shard',str(shard),'--shards','4','--batch','1','--memory-fraction','0.22']
        with job.with_suffix('.log').open('a') as log:
            p=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        history['launches'].append({'model':'robometer-4b','shard':shard,'gpu':shard,'pid':p.pid,
            'batch':1,'memory_fraction':.22,'started':time.time(),'reason':'bounded guarded overlap'})
        save(path,history)
        return p.pid


def run():
    EVAL.mkdir(parents=True,exist_ok=True);(EVAL/'jobs').mkdir(exist_ok=True)
    lock=(EVAL/'overlap-guard.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (EVAL/'overlap-guard.pid').write_text(str(os.getpid()))
    first=resources();initial_oom=first['oom_kill'];safe_since={i:None for i in range(4)}
    last_launch=0;started=time.time();restart_times={i:[] for i in range(4)}
    while True:
        now=time.time();sample=resources();training=read(BASE/'status.json',{})
        occupied={g for row in training.get('groups',[]) if row['phase']=='training' and alive(row.get('pid')) for g in row['gpus']}
        jobs={};actions={};history=read(EVAL/'early-launch.json',{'launches':[]})
        for gpu in range(4):
            name=f'robometer-4b-group1-{gpu}'
            job=read(EVAL/'jobs'/f'{name}.json',{})
            pids=[r['pid'] for r in history['launches'] if r['shard']==gpu and alive(r['pid'])]
            running=bool(pids) or alive(job.get('pid'))
            flag=EVAL/f'pause-{name}';reasons=pause_reasons(sample,gpu,initial_oom)
            if now-training.get('updated',0)>90:reasons.append('training_controller_status_stale')
            if gpu not in occupied:reasons.append('release_GPU_for_training_queue_transition')
            if (EVAL/'pause-all-early-evaluation').exists():reasons.append('manual_pause_all')
            if reasons:
                safe_since[gpu]=None;flag.touch();actions[gpu]={'pause_reasons':reasons}
            else:
                if safe_since[gpu] is None:safe_since[gpu]=now
                stable=now-safe_since[gpu]>=60
                if stable:
                    flag.unlink(missing_ok=True)
                    resumable=not job or job.get('status')=='paused_by_resource_guard'
                    if not running and resumable and now-last_launch>=60:
                        restart_times[gpu]=[t for t in restart_times[gpu] if now-t<3600]
                        if len(restart_times[gpu])<3:
                            pid=launch(gpu);last_launch=now;restart_times[gpu].append(now)
                            actions[gpu]={'launched_pid':pid}
                        else:actions[gpu]={'blocked':'three_launches_per_hour_limit'}
                else:actions[gpu]={'waiting_for_60_seconds_resource_stability':True}
            if not running and job and job.get('status')=='running':
                actions[gpu]={'blocked':'worker_disappeared_without_exit_status; inspect before restart'}
            jobs[gpu]=job
        save(BASE/'early-evaluation-status.json',{'evaluation_started':True,'stage':'guarded_parallel_baseline',
            'guard_pid':os.getpid(),'started':started,'updated':now,'resource_sample':sample,'actions':actions,
            'jobs':{i:{k:j.get(k) for k in ['status','pid','completed_episodes','completed_samples','input_blockers']} for i,j in jobs.items()},
            'training_processes_untouched':True,'next_models':'not launched by this guard'})
        if all(j.get('status') in DONE for j in jobs.values()):
            result=aggregate('robometer-4b')
            save(EVAL/'overlap-guard-completion.json',{'finished':time.time(),'complete':result['complete'],
                'missing_episodes':result['missing_episodes'],'overall':result['overall']})
            return
        time.sleep(10)


if __name__=='__main__':
    try:run()
    except BaseException:
        # On a guard failure, conservatively ask only the added workers to stop.
        for gpu in range(4):(EVAL/f'pause-robometer-4b-group1-{gpu}').touch()
        save(EVAL/'overlap-guard-error.json',{'error':traceback.format_exc(),'time':time.time()})
        raise
