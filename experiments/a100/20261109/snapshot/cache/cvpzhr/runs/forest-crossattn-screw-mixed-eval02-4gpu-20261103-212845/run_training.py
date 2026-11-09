"""Four-GPU restart with retained checkpoints and in-process validation every fifth epoch."""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

RUN = Path(__file__).resolve().parent
SETTINGS = json.loads((RUN/'settings.json').read_text())
CONFIG = json.loads((RUN/'config.json').read_text())
OUTPUT = Path(CONFIG['output_dir'])
STATE = dict(pid=os.getpid(), created_at=time.time(), physical_gpus=[0,1,2,3],
             epochs=3, expected_steps=1524, expected_validation_points=15,
             output=str(OUTPUT), batch_per_gpu=32, global_batch=128)


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def status(phase, **values):
    STATE.update(phase=phase, updated_at=time.time(), **values)
    write(RUN/'status.json', STATE)


def read_rows(path):
    if not path.exists():
        return []
    raw = path.read_text()
    return [json.loads(s) for s in raw[:raw.rfind('\n')+1].splitlines() if s.strip()]


def run_stage(phase, configuration):
    out = Path(json.loads(configuration.read_text())['output_dir'])
    environment = dict(os.environ, PYTHONPATH=SETTINGS['code'], CUDA_VISIBLE_DEVICES='0,1,2,3',
                       OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                       WANDB_DISABLED='true', NCCL_DEBUG='WARN')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
               '-m', 'mem_lite.train_stage3', '--config', str(configuration)]
    with (RUN/f'{phase}.log').open('w') as log:
        child = subprocess.Popen(command, cwd=SETTINGS['code'], env=environment, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    started, plotted = time.time(), 0
    status(phase, child_pid=child.pid, stage_started_at=started, last_train_record=None,
           last_validation_record=None, validation_points_completed=0)
    try:
        while child.poll() is None:
            training, validation = read_rows(out/'train.jsonl'), read_rows(out/'validation.jsonl')
            activity = 'training'
            if (out/'validation-status.json').exists():
                if json.loads((out/'validation-status.json').read_text())['phase']=='running':
                    activity = 'validation'
            status(phase, activity=activity, last_train_record=training[-1] if training else None,
                   last_validation_record=validation[-1] if validation else None,
                   validation_points_completed=len(validation))
            telemetry = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                                  '--format=csv,noheader'], text=True)
            with (RUN/'gpu.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(),phase=phase,readings=telemetry.strip().splitlines()))+'\n')
            if not training and time.time()-started>600:
                raise RuntimeError('No first update within 10 minutes')
            if phase=='training' and len(validation)>plotted:
                subprocess.run([sys.executable,str(RUN/'render_loss_curve.py')],check=True)
                plotted = len(validation)
            if phase=='preflight' and time.time()-started>900:
                raise RuntimeError('Preflight exceeded 15 minutes')
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(f'{phase} exited {child.returncode}; see {phase}.log')
        return time.time()-started
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def main():
    lock = (RUN/'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (RUN/'started.json').exists()
    ready = json.loads(Path(SETTINGS['cache_ready_report']).read_text())
    assert ready['status']=='ready' and ready['index_sha256']==SETTINGS['cache_index_sha256']
    assert CONFIG['init_checkpoint']==ready['initial_checkpoint']
    assert CONFIG['epochs']==3 and CONFIG['batch_size']==32 and CONFIG['validation_interval_epochs']==.2
    assert CONFIG['keep_only_latest'] is False
    digest = hashlib.sha256()
    with Path(CONFIG['init_checkpoint']).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            digest.update(block)
    assert digest.hexdigest()==SETTINGS['initial_checkpoint_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    write(RUN/'started.json',dict(time=time.time(),initial_checkpoint_sha256=digest.hexdigest(),
                                 cache_index_sha256=ready['index_sha256']))
    try:
        preflight_seconds = run_stage('preflight', RUN/'preflight.json')
        import torch
        probe = RUN/'preflight'
        state = torch.load(probe/'last.pt',map_location='cpu',weights_only=True)
        assert state['step']==2 and state['epoch']==0 and state['next_batch']==2
        assert state['config']['initialization']['path']==CONFIG['init_checkpoint']
        del state
        validation = read_rows(probe/'validation.jsonl')
        assert len(validation)==2 and [r['step'] for r in validation]==[1,2]
        expected = json.loads((RUN/'preflight.json').read_text())['validation']['expected_samples']
        assert all(r['num_samples']==expected and math.isfinite(r['loss']) for r in validation)
        assert sorted(p.name for p in probe.glob('*.pt'))==['last.pt','step-000001.pt','step-000002.pt']
        write(RUN/'preflight-result.json',dict(status='passed',seconds=preflight_seconds,
              optimizer_updates=2,validation_points=2,checkpoint_archives=2,
              resumed_training_after_validation=True,formal_initialization=CONFIG['init_checkpoint']))
        for p in probe.glob('*.pt'):
            p.unlink()
        OUTPUT.mkdir(parents=True,exist_ok=False)
        seconds = run_stage('training',RUN/'config.json')
        state = torch.load(OUTPUT/'last.pt',map_location='cpu',weights_only=True)
        assert state['step']==1524 and state['epoch']==3 and state['next_batch']==0
        del state
        validation = read_rows(OUTPUT/'validation.jsonl')
        expected_steps=[e*508+s for e in range(3) for s in (102,204,305,407,508)]
        assert [r['step'] for r in validation]==expected_steps
        assert all(r['num_samples']==21806 and math.isfinite(r['loss']) for r in validation)
        assert sorted(p.name for p in OUTPUT.glob('step-*.pt'))==[f'step-{s:06d}.pt' for s in expected_steps]
        subprocess.run([sys.executable,str(RUN/'render_loss_curve.py')],check=True)
        report=dict(status='completed',epochs=3,steps=1524,validation_points=15,
                    initial_checkpoint=CONFIG['init_checkpoint'],last_checkpoint=str(OUTPUT/'last.pt'),
                    checkpoint_archives=[r['checkpoint'] for r in validation],
                    training_and_validation_seconds=seconds,final_validation=validation[-1],
                    best_validation=min(validation,key=lambda r:r['loss']))
        write(OUTPUT/'report.json',report)
        write(RUN/'report.json',report)
        status('completed',completed_at=time.time(),validation_points_completed=15,report=str(OUTPUT/'report.json'))
    except Exception as exc:
        write(RUN/'failure.json',dict(error=str(exc),traceback=traceback.format_exc(),time=time.time()))
        status('failed',error=str(exc))
        raise


if __name__=='__main__':
    main()
