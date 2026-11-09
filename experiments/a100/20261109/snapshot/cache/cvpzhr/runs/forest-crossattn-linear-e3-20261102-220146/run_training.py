"""Authorized fresh three-epoch training; final-only checkpoint and stateful validation."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

RUN=Path(__file__).resolve().parent
SETTINGS=json.loads((RUN/'settings.json').read_text())
CODE=Path(SETTINGS['code']);OUT=Path(SETTINGS['output']);GPUS=[1,2,3]
STATE=dict(pid=os.getpid(),created_at=time.time(),physical_gpus=GPUS,world_size=3,
           training_authorized=True,shutdown_authorized=False,code=str(CODE),output=str(OUT),
           checkpoint_format='mem-lite-streamforest-crossattn-linear-v1',epochs=3,
           video_batch_per_gpu=4,global_video_batch=12,sequence_window=8,
           gradient_checkpointing=True,steps_per_epoch=72,expected_steps=216,
           train_samples=126441,validation_samples=32164)


def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def status(phase,**values):
    STATE.update(phase=phase,updated_at=time.time(),**values);write(RUN/'status.json',STATE)


def gpu_sample():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in raw.strip().splitlines():
        i,u,util,used=line.split(', ');gpus.append(dict(index=int(i),uuid=u,util=int(util),memory_mib=int(used)))
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(time=time.time(),gpus=gpus,processes=[line.split(', ') for line in raw.splitlines() if line])


def await_idle(stage):
    status('waiting_for_idle_gpu',next_stage=stage)
    clear=0
    while clear<2:
        sample=gpu_sample();used={p[0] for p in sample['processes']}
        ready=all(g['uuid'] not in used and g['util']<=5 and g['memory_mib']<1024 for g in sample['gpus'] if g['index'] in GPUS)
        clear=clear+1 if ready else 0
        if clear<2:time.sleep(15)


def run_stage(stage,args):
    await_idle(stage)
    env={**os.environ,'PYTHONPATH':str(CODE),'CUDA_VISIBLE_DEVICES':'1,2,3','OMP_NUM_THREADS':'1',
         'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'PYTHONDONTWRITEBYTECODE':'1','NCCL_DEBUG':'WARN'}
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',*args]
    with (RUN/f'{stage}.log').open('w') as log:
        child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    started=time.time();status(stage,child_pid=child.pid,stage_started_at=started)
    if stage=='training':STATE['training_started_at']=started
    try:
        while child.poll() is None:
            free=shutil.disk_usage(OUT).free/2**30;sample=gpu_sample()
            with (RUN/'gpu.jsonl').open('a') as f:f.write(json.dumps(dict(phase=stage,**sample))+'\n')
            status(stage,disk_free_gib=free)
            if free<2:raise RuntimeError('Disk free below 2 GiB; stopping this task only')
            for gpu in sample['gpus']:
                if gpu['index'] in GPUS and sum(p[0]==gpu['uuid'] for p in sample['processes'])>1:
                    raise RuntimeError(f"Another GPU process appeared on GPU {gpu['index']}; stopping this task only")
            time.sleep(15)
        if child.returncode:raise RuntimeError(f'{stage} failed with exit {child.returncode}; see {stage}.log')
        return time.time()-started
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    with (RUN.parent/'forest-crossattn-linear-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (RUN/'started.json').exists(),'Already started; no automatic rerun'
        cfg=json.loads((RUN/'config.json').read_text())
        assert cfg['epochs']==3 and cfg['batch_size']==4 and cfg['gradient_accumulation_steps']==1
        assert cfg['sequence_window']==8 and cfg['model']['gradient_checkpointing']
        assert cfg['save_every_steps'] is None and cfg['save_every_epochs']==3 and cfg['keep_only_latest']
        assert cfg['max_steps'] is None and not OUT.exists()
        OUT.mkdir()
        write(RUN/'started.json',dict(time=time.time(),config_sha256=hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest()))
        try:
            training_seconds=run_stage('training',['-m','mem_lite.train_stage3','--config',str(RUN/'config.json')])
            import torch
            checkpoint=torch.load(OUT/'last.pt',map_location='cpu',weights_only=True)
            assert checkpoint['format']==STATE['checkpoint_format'] and checkpoint['epoch']==3
            assert checkpoint['step']==216 and checkpoint['next_batch']==0
            assert sorted(p.name for p in OUT.glob('*.pt'))==['last.pt']
            del checkpoint
            validation_seconds=run_stage('validation',[str(RUN/'validate.py')])
            rows=[json.loads(line) for line in (OUT/'train.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows]==list(range(1,217))
            assert sum(r['samples'] for r in rows)==3*126441
            metrics=json.loads((OUT/'validation-03/metrics.json').read_text())
            assert metrics['num_samples']==32164 and metrics['coverage_verified'] and metrics['same_samples_and_pt_as_forest']
            train=dict(first_loss=rows[0]['loss'],last_group_loss=rows[-1]['loss'],
                       weighted_mean_loss=math.fsum(r['loss']*r['samples'] for r in rows)/sum(r['samples'] for r in rows),
                       samples_seen=sum(r['samples'] for r in rows),
                       losses_and_gradients_finite=all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows),
                       max_grad_norm_before_clip=max(r['grad_norm'] for r in rows),
                       peak_allocated_gib=max(max(r['peak_allocated_gib_per_gpu']) for r in rows),
                       peak_reserved_gib=max(max(r['peak_reserved_gib_per_gpu']) for r in rows))
            assert train['losses_and_gradients_finite']
            report=dict(status='completed',epochs=3,steps=216,checkpoint=str(OUT/'last.pt'),
                        checkpoint_format=STATE['checkpoint_format'],world_size=3,physical_gpus=GPUS,
                        video_batch_per_gpu=4,sequence_window=8,gradient_checkpointing=True,
                        training_seconds=training_seconds,validation_seconds=validation_seconds,
                        training=train,validation=metrics,source=json.loads((RUN/'source.json').read_text()))
            comparisons={}
            for name,directory in [('forest','forest-proprimem-e3-20261031-0009'),('forest_cross','forest-crossattn-v2-e3-20261101-210436')]:
                baseline=json.loads((Path('/cache/cvpzhr/outputs')/directory/'validation-03/metrics.json').read_text())
                comparisons[name]=dict(mae_pp=baseline['mae_pp'],rmse_pp=baseline['rmse_pp'],
                     mae_difference_pp=metrics['mae_pp']-baseline['mae_pp'],rmse_difference_pp=metrics['rmse_pp']-baseline['rmse_pp'])
            report['baseline_comparison']=comparisons
            report['comparison_note']='Same original video split/PT; current training uses ordered video groups with state carry and different update frequency. Not a training-configuration-matched ablation. Evaluation is cached batch inference, not online latency.'
            write(OUT/'report.json',report)
            status('completed',completed_at=time.time(),training_seconds=training_seconds,validation_seconds=validation_seconds)
        except BaseException as error:
            write(RUN/'failure.json',dict(error=str(error),traceback=traceback.format_exc(),time=time.time()))
            status('failed',error=str(error));raise


if __name__=='__main__':main()
