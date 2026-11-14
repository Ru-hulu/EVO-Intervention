"""Isolated five-model Recover validation; original datasets and training untouched."""
from pathlib import Path
from experiments.recover_finetune_20261114.common import read, rows, save, sha, tensor_digest

ROOT = Path('/cache/cvpzhr/setup/recover-five-validation-20261114')
OUT = Path('/cache/cvpzhr/evaluations/recover-five-validation-20261114')
DATA = Path('/cache/cvpzhr/datasets/ProPriMem/Recover')
IMAGES = Path('/cache/cvpzhr/datasets/recover-five-validation-images-20261114')
TRAIN = Path('/cache/cvpzhr/setup/recover-five-finetune-20261114')
PROC = Path('/cache/cvpzhr/setup/procvlm-recover-10-progress-only-e1-20261114')
FOREST = Path('/cache/cvpzhr/setup/forest-recover-10-legacy-e1-20261114')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
MODELS = ('procvlm-2b', 'forest', 'robobrain25-4b', 'dopamine-3b', 'robometer-4b')


def result_path(model, key):
    return OUT/model/'episodes'/(key.replace('/', '__')+'.json')


def select(tasks, seed=42):
    import random
    rng = random.Random(seed)
    chosen = {}
    for task in sorted(tasks, key=lambda t: t['task_id']):
        assert not set(task['train']) & set(task['val'])
        assert len(task['val']) == len(set(task['val'])) >= 2
        chosen[task['task_id']] = sorted(rng.sample(sorted(task['val']), 2))
    return chosen


def frame_inputs(row):
    """Inference view deliberately omits PT and annotations."""
    return dict(key=row['key'], task=row['task_prompt'], frames=[{'frame_index':i, 'image': [
        str(IMAGES/row['task_id']/row['sample_id']/f'{i:08d}-{v}.png') for v in range(3)]}
        for i in row['frame_indices']])


def host_sample():
    from experiments.prm_five_groups_20261114.overlap_guard import resources
    r = resources()
    stat = dict((k, int(v)) for k, v in (line.split() for line in
        Path('/sys/fs/cgroup/memory/memory.stat').read_text().splitlines()))
    clean = max(0, stat['total_cache']-sum(stat.get(k, 0) for k in
        ('total_shmem', 'total_mapped_file', 'total_dirty', 'total_writeback', 'total_unevictable')))
    r['nonreclaimable'] = max(r['resident_and_shmem'], r['memory_usage']-clean)
    return r


def safe(r, gpu, initial_oom):
    g = r['gpus'][gpu]
    return (r['resident_and_shmem'] < .65*r['memory_limit'] and
        r['nonreclaimable'] < .85*r['memory_limit'] and not r['under_oom'] and
        r['oom_kill'] == initial_oom and r['disk_free'] > 100*2**30 and
        g['temperature'] < 80 and g['total_mib']-g['used_mib'] > 16384)
