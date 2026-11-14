"""Audited State-only maintenance; scientific configuration remains immutable."""
import json
from pathlib import Path
import time

from .evaluation_priority import matches
from .prepare_cache import sha


def request(base):
    path = base / 'state-acceleration/request.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('status') not in {'testing', 'resume_requested'}:
        return None
    assert value.get('user_authorized') is True
    assert value['suite'] == 'State' and value['gpus'] == [0, 1]
    assert matches(value['supervisor']), 'Acceleration supervisor is no longer alive'
    return value


def permits_existing_evaluation(base, gpu):
    """Maintenance replaces training, never adds a third process or stops eval."""
    try:
        value = request(base)
        return value is not None and gpu in value['gpus']
    except (AssertionError, OSError, ValueError, KeyError):
        return False


def validate_resume(base, value):
    import torch
    from .prepare_state_acceleration import ENV, ORIGINAL
    path = base / 'State/training.config.json'
    assert sha(path) == value['config_sha256']
    checkpoint = Path(value['checkpoint'])
    assert checkpoint.parent == base / 'state-acceleration'
    assert sha(checkpoint) == value['checkpoint_sha256']
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = json.loads(path.read_text())
    assert ckpt['format'] == 'mem-lite-streamforest-v1'
    assert ckpt['epoch'] == 0 and 0 < ckpt['step'] < ckpt['config']['total_optimizer_steps']
    assert ckpt['next_batch'] == ckpt['step'] * config['gradient_accumulation_steps']
    assert len(ckpt['rng_by_rank']) == 2 and ckpt['optimizer']['state']
    assert ckpt['scheduler']['last_epoch'] == ckpt['step']
    for key, expected in config.items():
        assert ckpt['config'][key] == expected, f'Changed training configuration: {key}'
    assert config['run_protocol']['gpus'] == [0, 1]
    assert config['batch_size'] == 1 and config['gradient_accumulation_steps'] == 9
    assert config['model']['gradient_checkpointing'] is True
    if value['backend'] == 'accelerated':
        proof = json.loads((base / 'state-acceleration/verification.json').read_text())
        assert proof['status'] == 'passed' and proof['checkpoint_sha256'] == value['checkpoint_sha256']
        assert proof['numerical_passed'] and proof['two_rank_smoke_passed'] and proof['memory_passed']
        assert proof['config_sha256'] == value['config_sha256']
        assert proof.get('native_gated_norm',False)==value.get('native_gated_norm',False)
        python = ENV / 'bin/python'
    else:
        assert value['backend'] == 'original'
        assert value['reason'] == 'acceleration_validation_failed_restore_original_training'
        python = ORIGINAL / 'bin/python'
    return path, config, str(checkpoint), str(python)


def mark_resumed(base, value, pid):
    from .prepare_cache import save
    save(base / 'state-acceleration/request.json', dict(value, status='resumed',
         resumed_pid=pid, resumed_at=time.time()))
