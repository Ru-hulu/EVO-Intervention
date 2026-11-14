"""Adopt the running Order trainer and start Count on the newly idle GPU pair."""
import fcntl
import os
import subprocess
import time
import traceback
from .common import *
from .queue import host_health, idle


def process(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(state=fields[0],start_ticks=int(fields[19]))
    except FileNotFoundError:
        return None


def alive(pid,start_ticks):
    current=process(pid)
    return bool(current and current['start_ticks']==start_ticks and current['state'] not in ('Z','X'))


def verify(cfg):
    report=read(Path(cfg['setup_dir'])/'formal/verification.json')
    assert report['status']=='passed' and not report['pilot'] and report['epoch']==1
    assert report['suite']==cfg['suite'] and report['expected_samples']==cfg['samples']
    assert report['expected_episodes']==35 and report['train_input_sha256']==cfg['train_sha256']
    assert report['fresh_official_initialization'] and report['replicas_equal']
    assert sha(cfg['train_data'])==cfg['train_sha256']
    checkpoint=Path(report['checkpoint'])
    assert read(checkpoint/'run_config.json')==cfg
    assert sha(checkpoint/'trainable-state.pt')==report['trainable_state_sha256']
    return report


def main():
    home=ROOT/'robometer-4b';request=read(home/'parallel-count-request.json')
    lock=(home/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not alive(request['old_controller_pid'],request['old_controller_start_ticks'])
    order_pid=request['order_pid'];order_ticks=request['order_start_ticks']
    assert alive(order_pid,order_ticks),'Order trainer must continue without interruption'
    assert idle([2,3]),'Leave other GPU jobs untouched'
    order_cfg=read(home/'Order/config.json')
    assert sha(home/'Order/config.json')==request['order_config_sha256']
    assert sha(home/'Order/formal/launch.json')==request['order_launch_sha256']
    cfg=read(home/'Count/reference-config.json')
    choice=read(home/'batch-choice.json')['choice'];micro=choice['microbatch']
    cfg.update(gpus=[2,3],microbatch=micro,accumulation=cfg['effective_batch']//(2*micro))
    assert cfg['suite']=='Count' and cfg['samples']==4315 and cfg['epochs']==1
    assert not (home/'Count/formal/launch.json').exists()
    assert not Path(cfg['output_dir']).exists()
    baseline=read(home/'resource-baseline.json')
    health=host_health()
    assert not health['under_oom'] and health['oom_kill']==baseline['oom_kill']
    assert health['resident']<.8*health['limit']
    assert not (home/'resource-stop.json').exists()
    config_path=home/'Count/config.json';save(config_path,cfg)
    folder=home/'Count/formal';folder.mkdir(parents=True,exist_ok=True)
    command=[cfg['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        '-m','experiments.soc35_finetune_20261114.train','--config',str(config_path)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',NCCL_DEBUG='WARN',
        TRITON_CACHE_DIR=str(home/'count-triton-cache'))
    with (folder/'train.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(folder/'launch.json',dict(pid=child.pid,command=command,gpus=[2,3],started=time.time(),
        config_sha256=sha(config_path),parallel_with='Order',fresh_official_initialization=True))
    jobs={
        'Order':dict(pid=order_pid,start_ticks=order_ticks,gpus=[0,1],status='training',adopted=True),
        'Count':dict(pid=child.pid,start_ticks=process(child.pid)['start_ticks'],gpus=[2,3],status='training',adopted=False),
    }
    status=read(home/'status.json')
    assert [x['suite'] for x in status['completed']]==['State']
    status.update(controller_pid=os.getpid(),previous_controller_pid=request['old_controller_pid'],
        mode='parallel_order_count',suite='Order+Count',gpus=[0,1,2,3],active_pid=None,
        active_jobs=jobs,rescheduled=time.time(),count_removed_from_sequential_queue=True)
    configs={'Order':order_cfg,'Count':cfg}
    while True:
        for suite,job in jobs.items():
            if job['status']!='training':continue
            ended=child.poll() is not None if suite=='Count' else not alive(job['pid'],job['start_ticks'])
            if not ended:continue
            output=Path(configs[suite]['setup_dir'])/'formal'
            try:
                if suite=='Count':assert child.returncode==0,f'Count trainer returned {child.returncode}'
                report=verify(configs[suite])
                status['completed'].append(dict(suite=suite,checkpoint=report['checkpoint'],verification=report))
                job.update(status='completed',checkpoint=report['checkpoint'],finished=time.time())
                save(output/'exit.json',dict(returncode=child.returncode if suite=='Count' else None,
                    adopted_process=suite=='Order',completion_verified=True,finished=time.time()))
            except Exception:
                job.update(status='failed',error=traceback.format_exc(),finished=time.time())
                save(output/'exit.json',dict(returncode=child.returncode if suite=='Count' else None,
                    adopted_process=suite=='Order',completion_verified=False,error=job['error'],finished=time.time()))
        active=[suite for suite,job in jobs.items() if job['status']=='training']
        status.update(updated=time.time(),suite='+'.join(active) or 'Count',active_jobs=jobs)
        if not active:
            status.update(status='completed' if all(j['status']=='completed' for j in jobs.values()) else 'failed',finished=time.time())
            if status['status']=='failed':status['error']={s:j['error'] for s,j in jobs.items() if j['status']=='failed'}
        save(home/'status.json',status)
        if not active:break
        health=host_health()
        if health['under_oom'] or health['oom_kill']!=baseline['oom_kill'] or health['resident']>.8*health['limit']:
            save(home/'resource-stop.json',dict(reason='Host memory pressure',health=health))
        time.sleep(5)


if __name__=='__main__':
    try:main()
    except BaseException:
        home=ROOT/'robometer-4b'
        save(home/'parallel-controller-failure.json',dict(error=traceback.format_exc(),updated=time.time()))
        raise
