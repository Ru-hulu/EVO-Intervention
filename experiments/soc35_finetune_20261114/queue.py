"""Two detached model queues with bounded batch pilots and independent suites."""
import argparse
import copy
import fcntl
import os
import statistics
import subprocess
import time
import traceback
from .common import *


def idle(gpus):
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    ids=[]
    for line in raw.splitlines():
        i,u,m=[x.strip() for x in line.split(',')]
        if int(i) in gpus:
            if int(m)>=1000:return False
            ids.append(u)
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return len(ids)==2 and not any(u in running for u in ids)


def host_health():
    base=Path('/sys/fs/cgroup/memory')
    stat=dict((k,int(v)) for k,v in map(str.split,(base/'memory.stat').read_text().splitlines()))
    limit=int((base/'memory.limit_in_bytes').read_text())
    resident=stat['total_rss']+stat.get('total_shmem',0)
    oom=dict((k,int(v)) for k,v in map(str.split,(base/'memory.oom_control').read_text().splitlines()))
    return dict(resident=resident,limit=limit,under_oom=oom.get('under_oom',0),oom_kill=oom.get('oom_kill',0))


def wait_idle(gpus,timeout=120):
    until=time.time()+timeout
    while not idle(gpus):
        assert time.time()<until,'CUDA contexts did not release; no other process will be killed'
        time.sleep(2)


def launch(cfg,pilot,status_path,baseline_oom):
    assert idle(cfg['gpus'])
    config_path=Path(cfg['setup_dir'])/'config.json';save(config_path,cfg)
    folder=Path(cfg['setup_dir'])/('pilot' if pilot else 'formal');folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'launch.json').exists(),'Refuse duplicate launch'
    command=[cfg['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        '-m','experiments.soc35_finetune_20261114.train','--config',str(config_path)]
    if pilot:command+=['--pilot']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpus'])),OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',NCCL_DEBUG='WARN',
        TRITON_CACHE_DIR=str(ROOT/cfg['name']/'triton-cache'))
    with (folder/'train.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(folder/'launch.json',dict(pid=child.pid,command=command,gpus=cfg['gpus'],started=time.time(),config_sha256=sha(config_path)))
    while child.poll() is None:
        status=read(status_path);status.update(active_pid=child.pid,updated=time.time());save(status_path,status)
        health=host_health()
        if health['under_oom'] or health['oom_kill']!=baseline_oom or health['resident']>.8*health['limit']:
            save(ROOT/cfg['name']/'resource-stop.json',dict(reason='Host memory pressure',health=health))
        time.sleep(5)
    save(folder/'exit.json',dict(returncode=child.returncode,finished=time.time()))
    wait_idle(cfg['gpus'])
    return child.returncode,folder


def main(name):
    home=ROOT/name;home.mkdir(exist_ok=True,parents=True)
    lock=(home/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path=home/'status.json';assert not status_path.exists(),'Inspect existing queue before restart'
    plan=read(ROOT/'plan.json');configs=plan['configs'][name];gpus=GPUS[name]
    assert idle(gpus),'Assigned GPUs are occupied; leave other tasks alone'
    baseline=host_health();assert not baseline['under_oom'] and baseline['resident']<.8*baseline['limit']
    save(home/'resource-baseline.json',baseline)
    status=dict(status='calibrating',model=name,suite='State',gpus=gpus,controller_pid=os.getpid(),
        started=time.time(),updated=time.time(),completed=[],order=SUITES)
    save(status_path,status)
    candidates=[8,4] if name=='robometer-4b' else [16,8]
    trials=[]
    for micro in candidates:
        cfg=copy.deepcopy(configs['State']);cfg.update(microbatch=micro,
            accumulation=cfg['effective_batch']//(2*micro),setup_dir=str(home/'trials'/f'b{micro}'))
        code,folder=launch(cfg,True,status_path,baseline['oom_kill'])
        assert not (home/'resource-stop.json').exists(),'Host resource protection triggered'
        if code==0:
            report=read(folder/'verification.json');assert report['status']=='passed' and report['pilot']
            elapsed=max(statistics.median(x[1:]) for x in report['pilot_step_seconds'])
            trials.append(dict(microbatch=micro,status='passed',samples_per_second=2*micro/elapsed,
                peak_reserved_gib=report['peak_reserved_gib'],verification=str(folder/'verification.json')))
        else:
            log=(folder/'train.log').read_text().lower()
            assert 'out of memory' in log or 'gpu headroom' in log,f'Non-memory pilot failure: {folder}'
            trials.append(dict(microbatch=micro,status='memory_limit',log=str(folder/'train.log')))
        save(home/'trials.json',trials)
    good=[t for t in trials if t['status']=='passed'];assert good,'No valid pilot configuration'
    choice=max(good,key=lambda t:t['samples_per_second'])
    save(home/'batch-choice.json',dict(choice=choice,policy='Best measured post-warmup throughput among bounded native-input candidates',
        fixed_effective_batch=configs['State']['effective_batch'],pilot_weights_discarded=True))
    for suite in SUITES:
        cfg=copy.deepcopy(configs[suite]);micro=choice['microbatch']
        cfg.update(microbatch=micro,accumulation=cfg['effective_batch']//(2*micro))
        status=read(status_path);status.update(status='training',suite=suite,microbatch=micro,
            effective_batch=cfg['effective_batch'],updated=time.time());save(status_path,status)
        code,folder=launch(cfg,False,status_path,baseline['oom_kill'])
        assert code==0,f'{suite} training failed; preserve artifacts: {folder}'
        report=read(folder/'verification.json')
        assert report['status']=='passed' and not report['pilot'] and report['epoch']==1
        assert report['expected_samples']==cfg['samples'] and report['expected_episodes']==35
        assert sha(cfg['train_data'])==cfg['train_sha256']
        status=read(status_path);status['completed'].append(dict(suite=suite,checkpoint=report['checkpoint'],verification=report))
        status.update(active_pid=None,updated=time.time());save(status_path,status)
    status.update(status='completed',finished=time.time());save(status_path,status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=list(GPUS));args=parser.parse_args()
    try:main(args.model)
    except BaseException:
        p=ROOT/args.model/'status.json';d=read(p) if p.exists() else {}
        d.update(status='failed',error=traceback.format_exc(),updated=time.time());save(p,d)
        raise
