"""Retain the latest evaluated checkpoint; never prune before successful validation."""
import json
import os
from pathlib import Path
import re

import torch


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def checked_name(name, step):
    if not re.fullmatch(r'step-\d{6,}\.pt', name) or name != f'step-{step:06d}.pt':
        raise ValueError('Checkpoint filename does not match its step')


def verify_checkpoint(path, step):
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('format') != 'mem-lite-streamforest-v1' or saved.get('step') != step:
        raise ValueError('Saved checkpoint has the wrong format or step')
    if set(saved['model']) != {'m3_lora', 'history_projection', 'progress_head'}:
        raise ValueError('Saved checkpoint is not the original Forest model')
    del saved


def save_rolling_checkpoint(state, output_dir, name):
    output = Path(output_dir)
    step = state['step']
    checked_name(name, step)
    path = output/name
    if state['config'].get('init_checkpoint') and path.resolve() == Path(state['config']['init_checkpoint']).resolve():
        raise ValueError('Cannot overwrite the initialization checkpoint')
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    verify_checkpoint(path, step)
    with (output/'checkpoint-writes.jsonl').open('a') as stream:
        stream.write(json.dumps({'name':name, 'step':step, 'bytes':path.stat().st_size})+'\n')
    write_json(output/'pending-checkpoint.json', {'checkpoint':name, 'step':step, 'evaluation_completed':False})


def commit_evaluated_checkpoint(output_dir, name, step):
    output = Path(output_dir)
    checked_name(name, step)
    path = output/name
    verify_checkpoint(path, step)
    metrics = json.loads((output/'validation'/f'step-{step:06d}'/'metrics.json').read_text())
    status = json.loads((output/'validation-status.json').read_text())
    if (metrics['step'] != step or Path(metrics['checkpoint']).resolve() != path.resolve()
            or status['phase'] != 'completed' or status['step'] != step):
        raise ValueError('The new checkpoint has not completed validation')
    entries = [json.loads(line) for line in (output/'checkpoint-writes.jsonl').read_text().splitlines()]
    # Switch the latest pointer atomically; it occupies no duplicate model storage.
    link = output/'last.pt.tmp'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(name)
    link.replace(output/'last.pt')
    removed = []
    for entry in entries:
        checked_name(entry['name'], entry['step'])
        old = output/entry['name']
        if entry['step'] < step and old.exists():
            if old.is_symlink() or old.resolve().parent != output.resolve():
                raise ValueError('Refusing to delete a checkpoint outside this run')
            old.unlink()
            removed.append(entry['name'])
    write_json(output/'checkpoint-retention.json', {'latest':name, 'step':step,
               'policy':'latest successfully evaluated checkpoint only', 'removed':removed})
    write_json(output/'pending-checkpoint.json', {'checkpoint':name, 'step':step, 'evaluation_completed':True})
