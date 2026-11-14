"""Sequential detached queue; no retries of unknown failures, no competing GPU jobs."""
import fcntl
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from .common import *


def resources():
    from experiments.prm_five_groups_20261114.overlap_guard import resources as existing
    return existing()


def idle_cards():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,temperature.gpu',
                                 '--format=csv,noheader,nounits'],text=True)
    selected=[]
    for row in raw.strip().splitlines():
        index,uuid,used,temp=[v.strip() for v in row.split(',')]
        if int(index) in (0,1):
            if float(used)>=1000 or float(temp)>=75:return False
            selected.append(uuid)
    running=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'],text=True)
    return len(selected)==2 and not any(uuid in running for uuid in selected)


def host_safe():
    # Cgroup accounting: inspect true RSS/shmem as well as non-reclaimable working set.
    r=resources()
    stat={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/memory/memory.stat').read_text().splitlines())}
    # Clean active file pages are reclaimable too; usage-inactive_file alone overcounts them.
    # Conservatively exclude shmem, mapped, dirty, writeback and unevictable bytes from reclaimable cache.
    clean=max(0,stat['total_cache']-sum(stat.get(k,0) for k in
        ('total_shmem','total_mapped_file','total_dirty','total_writeback','total_unevictable')))
    r['conservative_reclaimable_file_cache']=clean
    r['conservative_nonreclaimable']=max(r['resident_and_shmem'],r['memory_usage']-clean)
    save(ROOT/'resources.json',r)
    limit=r['memory_limit']
    initial=read(ROOT/'resource-baseline.json')
    return (r['resident_and_shmem']<.65*limit and r['conservative_nonreclaimable']<.85*limit
            and not r['under_oom'] and r['oom_kill']==initial['oom_kill'] and r['disk_free']>100*2**30
            and all(r['gpus'][i]['temperature']<80 for i in (0,1)))


def launch(config_path,pilot):
    cfg=read(config_path);name=cfg['name']
    folder=ROOT/name/('pilot' if pilot else 'formal');folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'launch.json').exists(),'Inspect existing launch before any restart'
    python=cfg['python']
    module='train'
    cmd=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
         '-m',f'experiments.recover_finetune_20261114.{module}','--config',str(config_path)]
    if pilot:cmd.append('--pilot')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
        TRITON_CACHE_DIR=str(ROOT/'triton-cache'))
    env['NCCL_DEBUG']='WARN'
    env.pop('PRM_CUDA_MEMORY_FRACTION',None);env.pop('FOREST_NATIVE_GATED_NORM',None)
    with (folder/'train.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(folder/'launch.json',dict(pid=process.pid,command=cmd,started=time.time(),config_sha256=sha(config_path)))
    while process.poll() is None:
        time.sleep(10)
        status=read(ROOT/'status.json');status.update(active_pid=process.pid,updated=time.time());save(ROOT/'status.json',status)
        if not host_safe():
            save(ROOT/'resource-stop.json',dict(reason='Host resource threshold reached',timestamp=time.time()))
            # HF workers cooperate at the next synchronized optimizer boundary.
    save(folder/'exit.json',dict(returncode=process.returncode,finished=time.time()))
    return process.returncode


def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'status.json').exists(),'Do not relaunch an existing queue without explicit audit'
    (ROOT/'controller.pid').write_text(str(os.getpid())+'\n')
    plan=read(ROOT/'plan.json');assert plan['order']==ORDER
    save(ROOT/'resource-baseline.json',resources())
    save(ROOT/'status.json',dict(status='starting',order=ORDER,completed=[],gpus=[0,1],updated=time.time()))
    for name in ORDER:
        config_path=Path(plan['configs'][name]);assert sha(config_path)==plan['config_sha256'][name]
        status=read(ROOT/'status.json');status.update(status='waiting_for_idle_cards',model=name,updated=time.time());save(ROOT/'status.json',status)
        stable=None
        while True:
            okay=idle_cards() and host_safe()
            if okay:
                stable=stable or time.time()
                if time.time()-stable>=30:break
            else:stable=None
            time.sleep(5)
        status.update(status='pilot',updated=time.time());save(ROOT/'status.json',status)
        assert launch(config_path,True)==0,f'{name} pilot failed; preserve logs, no blind retry'
        result=read(ROOT/name/'pilot/verification.json');assert result['status']=='passed'
        assert result['peak_reserved_gib']<63,'Insufficient GPU memory reserve'
        assert not (ROOT/'resource-stop.json').exists()
        # The pilot process and all its CUDA contexts must have exited before fresh training.
        limit=time.time()+120
        while not idle_cards():
            assert time.time()<limit,'CUDA context persisted after pilot';time.sleep(2)
        assert host_safe()
        status.update(status='training',updated=time.time());save(ROOT/'status.json',status)
        assert launch(config_path,False)==0,f'{name} formal run failed; preserve checkpoints'
        report=read(ROOT/name/'formal/verification.json');assert report['status']=='passed'
        status['completed'].append(dict(model=name,verification=report));status.update(updated=time.time())
        save(ROOT/'status.json',status)
    status.update(status='completed',active_pid=None,updated=time.time());save(ROOT/'status.json',status)


if __name__=='__main__':
    try:main()
    except Exception:
        detail=traceback.format_exc();print(detail,flush=True)
        status=read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        status.update(status='failed',error=detail,updated=time.time());save(ROOT/'status.json',status)
        raise
