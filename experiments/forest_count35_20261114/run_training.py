"""Dedicated one-epoch Count runner, with no other-suite scheduling."""
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path('/cache/cvpzhr/setup/forest-count-35-legacy-e1-20261114')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'


def save(value):
    tmp = ROOT / 'training-status.tmp'
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(ROOT / 'training-status.json')


def main():
    assert not (ROOT / 'TRAINING_SUPERSEDED.json').exists(), 'User superseded this run'
    lock = (ROOT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((ROOT / 'training.config.json').read_text())
    assert config['epochs'] == 1 and config['run_protocol']['suite'] == 'Count'
    assert config['batch_size'] == 4 and config['gradient_accumulation_steps'] == 1
    assert config['model']['gradient_checkpointing'] is False
    assert config['lr_schedule'] == 'constant'
    output = Path(config['output_dir'])
    assert not output.exists(), 'Refuse to overwrite an existing run'
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    assert importlib.metadata.version('flash-linear-attention') == '0.3.2'
    assert importlib.metadata.version('fla-core') == '0.3.2'
    assert callable(qwen.chunk_gated_delta_rule) and qwen.causal_conv1d_fn is None
    assert callable(qwen.FusedRMSNormGated)
    environment_report = dict(
        packages={n: importlib.metadata.version(n) for n in
                  ('torch', 'transformers', 'triton', 'flash-linear-attention', 'fla-core', 'einops')},
        linear_attention_acceleration=True, causal_conv1d_installed=False,
        environment=PYTHON, legacy_environment_restored=True)
    (ROOT / 'environment.json').write_text(json.dumps(environment_report, indent=2) + '\n')
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', TRITON_CACHE_DIR=str(ROOT / 'triton-cache'))
    environment.pop('FOREST_NATIVE_GATED_NORM', None)
    environment.pop('PRM_CUDA_MEMORY_FRACTION', None)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    for row in gpu_rows.strip().splitlines():
        index, used = map(int, row.split(','))
        if index in (0, 1):
            assert used < 1000, f'GPU {index} occupied; will not disturb another task'
    command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               '-m', 'mem_lite.train_stage3', '--config', str(ROOT / 'training.config.json')]
    started = time.time()
    with (ROOT / 'train.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT / 'code', env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    expected_steps = math.ceil(math.ceil(config['run_protocol']['train_samples'] / 2) / 4)
    status = dict(status='running', pid=process.pid, runner_pid=os.getpid(), started=started,
        command=command, output_dir=str(output), suite='Count', epochs=1, gpus=[0, 1],
        expected_optimizer_steps=expected_steps, expected_samples=config['run_protocol']['train_samples'],
        no_old_weights_loaded=True, no_other_suites_scheduled=True)
    save(status)
    code = process.wait()
    status.update(returncode=code, finished=time.time())
    if code:
        status['status'] = 'failed'
        save(status)
        return code
    import torch
    checkpoint_path = output / 'epoch-01.pt'
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert ckpt['epoch'] == 1 and ckpt['next_batch'] == 0 and ckpt['step'] == expected_steps
    assert ckpt['scheduler'] is None and len(ckpt['rng_by_rank']) == 2
    assert ckpt['optimizer']['state'], 'Final checkpoint must preserve optimizer state'
    assert ckpt['config']['data_signature'] == config['data_signature']
    for state in ckpt['model'].values():
        assert all(torch.isfinite(value).all() for value in state.values())
    audits = [json.loads((output / f'initialization-rank{rank}-fresh.json').read_text()) for rank in (0, 1)]
    assert all(a['fresh_language_lora'] and not a['resume'] for a in audits)
    assert audits[0]['initial_trainable_sha256'] == audits[1]['initial_trainable_sha256']
    status.update(status='completed', checkpoint=str(checkpoint_path),
                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  final_checkpoint_verified=True, fresh_initialization_verified=True)
    save(status)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        detail = traceback.format_exc()
        print(detail, flush=True)
        save(dict(status='failed', error=detail, timestamp=time.time()))
        raise
