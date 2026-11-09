"""一次性的三轮训练与最终验证编排；只保存最终检查点，不自动重训。"""
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

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')
CODE='/cache/cvpzhr/code/forest-train3-20261031-0009'
CONFIG=RUN/'config.json'
STATE=dict(pid=os.getpid(),created_at=time.time(),training_authorized=True,shutdown_authorized=False)


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def status(phase,**values):
    STATE.update(phase=phase,updated_at=time.time(),**values);write_json(RUN/'status.json',STATE)


def gpu_sample():
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    rows=[list(map(int,row.split(','))) for row in query.strip().splitlines()]
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip().splitlines()
    return dict(time=time.time(),gpus=rows,compute_pids=pids)


def idle():
    sample=gpu_sample()
    return len(sample['gpus'])==4 and not sample['compute_pids'] and all(row[1]<=5 and row[2]<1024 for row in sample['gpus'])


def await_idle(stage):
    status('waiting_for_idle_gpu',next_stage=stage)
    while True:
        if idle():
            time.sleep(15)
            if idle():return
        time.sleep(30)


def run_stage(stage,args,output):
    await_idle(stage)
    env=os.environ.copy();env.update(PYTHONPATH=CODE,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='4')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',*args]
    with (RUN/f'{stage}.log').open('w') as log:
        child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status(stage,child_pid=child.pid,stage_started_at=time.time())
    try:
        while child.poll() is None:
            free=shutil.disk_usage(output).free/2**30
            with (RUN/'gpu.jsonl').open('a') as stream:stream.write(json.dumps(gpu_sample())+'\n')
            status(stage,free_gib=free)
            if free<2:raise RuntimeError('Disk free below 2 GiB: stopped only this job')
            time.sleep(20)
        if child.returncode:raise RuntimeError(f'{stage} failed with return code {child.returncode}; see {stage}.log')
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'started.json').exists():raise RuntimeError('This run has already started; no automatic restart')
        config=json.loads(CONFIG.read_text());output=Path(config['output_dir'])
        assert config['epochs']==3 and config['save_every_epochs']==3 and config['keep_only_latest']
        assert config['save_every_steps'] is None and config['max_steps'] is None
        assert not output.exists() or not any(output.iterdir()),'Output must be new'
        output.mkdir(parents=True,exist_ok=True)
        config_digest=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        expected_steps=3*math.ceil(math.ceil(126441/4)/config['batch_size'])
        write_json(RUN/'started.json',dict(started_at=time.time(),config_sha256=config_digest,expected_steps=expected_steps))
        STATE.update(output=str(output),epochs=3,expected_steps=expected_steps,batch_per_gpu=config['batch_size'],effective_batch=4*config['batch_size'])
        try:
            run_stage('training',['-m','mem_lite.train_stage3','--config',str(CONFIG)],output)
            import torch
            checkpoint=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            assert checkpoint['epoch']==3 and checkpoint['step']==expected_steps and checkpoint['next_batch']==0
            assert sorted(p.name for p in output.glob('*.pt'))==['last.pt']
            del checkpoint
            run_stage('validation',[str(RUN/'validate.py')],output)
            rows=[json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
            metrics=json.loads((output/'validation-03/metrics.json').read_text())
            report=dict(status='completed',epochs=3,steps=expected_steps,batch_per_gpu=config['batch_size'],
                        effective_batch=4*config['batch_size'],checkpoint=str(output/'last.pt'),
                        training=dict(first_logged_loss=rows[0]['loss'],last_logged_loss=rows[-1]['loss'],
                        mean_logged_step_loss=sum(r['loss'] for r in rows)/len(rows),logged_steps=len(rows),
                        gradients_all_finite=all(math.isfinite(r['grad_norm']) for r in rows),
                        max_logged_grad_norm=max(r['grad_norm'] for r in rows),
                        peak_allocated_gib=max(max(r['peak_allocated_gib_per_gpu']) for r in rows),
                        peak_reserved_gib=max(max(r['peak_reserved_gib_per_gpu']) for r in rows)),
                        validation=metrics,config_sha256=config_digest)
            write_json(output/'report.json',report)
            status('completed_pending_local_sync',completed_at=time.time())
        except Exception as exc:
            write_json(RUN/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
            status('failed',error=str(exc));raise


if __name__=='__main__':main()
