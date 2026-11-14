"""Immutable, model-specific input caches with original PT supervision.

This preparation does not choose an SFT response/loss for generative models.
In particular, absolute PT is NOT a VLAC critic target or synthetic Proc CoT.
"""
import json
from pathlib import Path
import random

from experiments.prm_five_groups_20261114.prepare_cache import sha, save

ROOT = Path('/cache/cvpzhr/setup/recover-six-model-caches-20261114')
OUT = Path('/cache/cvpzhr/datasets/recover-six-model-caches-5fps-20261114')
DATA = Path('/cache/cvpzhr/datasets/ProPriMem/Recover')
OLD = Path('/cache/cvpzhr/setup/nine-model-zeroshot-cleanval-voc-20261113')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
VIEWS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
MODELS = ['robometer-4b', 'procvlm-2b', 'robobrain25-4b', 'dopamine-3b', 'vlac-2b', 'forest']


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, records):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in records))
    tmp.replace(path)


def select(tasks, seed=42):
    rng = random.Random(seed); result = {}
    for task in sorted(tasks, key=lambda t:t['task_id']):
        train, val = task['train'], task['val']
        if len(train) != len(set(train)) or len(train) < 2 or set(train) & set(val):
            raise ValueError('Non-disjoint splits, duplicates or insufficient training data')
        result[task['task_id']] = sorted(rng.sample(sorted(train), 2))
    return result


def slots(model, i, n):
    """(sample index, camera index); only the two GRMs can see the endpoint."""
    if not 0 <= i < n:
        raise ValueError('Invalid sample position')
    if model == 'robometer-4b':
        import numpy as np
        idx = list(range(i+1))+[i]*(7-i) if i < 7 else np.rint(np.linspace(0,i,8)).astype(int).tolist()
        return [(j,0) for j in idx[:7]]+[(i,v) for v in range(3)]
    if model == 'procvlm-2b':
        return [(max(0,j),0) for j in range(i-7,i+1)]
    if model in ('robobrain25-4b','dopamine-3b'):
        return [(0,0),(n-1,0)]+[(0,v) for v in range(3)]+[(i,v) for v in range(3)]
    if model == 'vlac-2b':
        return [(max(0,i-1),0),(i,0)]
    if model == 'forest':
        return [(i,v) for v in range(3)]
    raise ValueError(model)


def epdir(row):
    return OUT/'shared'/row['task_id']/row['sample_id']


def source_checks(selection):
    for path, digest in selection['source_sha256'].items():
        assert sha(path) == digest, f'Source changed: {path}'


def pixel_path(model, row, index, view):
    return OUT/model/'pixels'/row['task_id']/row['sample_id']/f'{index:06d}-{view}.safetensors'
