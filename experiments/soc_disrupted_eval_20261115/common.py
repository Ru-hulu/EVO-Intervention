import json
import hashlib
from pathlib import Path
from experiments.prm_five_groups_20261114.prepare_cache import save, sha

ROOT = Path('/cache/cvpzhr/setup/soc-disrupted-eval-20261115')
OUT = Path('/cache/cvpzhr/evaluations/soc-disrupted-eval-20261115')
DATA = Path('/cache/cvpzhr/datasets/soc-two-per-task-perturbations-5fps-20261114')
DATA_SETUP = Path('/cache/cvpzhr/setup/soc-two-per-task-perturbations-20261114')
TRAIN = Path('/cache/cvpzhr/setup/soc35-procvlm-robometer-e1-20261114')
TRAIN_SELECTION = Path('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114/selection.json')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
MODELS = ('procvlm-2b', 'robometer-4b', 'forest')
SUITES = ('State', 'Order', 'Count')
SEEDS = (42, 43, 44)
CONDITIONS = tuple(f'{v}-seed{s}' for v in ('static_only', 'rewind_only') for s in SEEDS)


def read(path):
    return json.loads(Path(path).read_text())


def result_path(model, condition, key):
    relative = Path(key)
    assert not relative.is_absolute() and '..' not in relative.parts
    return OUT/condition/model/'episodes'/relative.with_suffix('.json')


def code_hashes():
    base = Path(__file__).parents[2]
    paths = []
    for folder in ('experiments', 'mem_lite', 'qwen_training'):
        paths += list((base/folder).rglob('*.py'))
    return {str(p.relative_to(base)): sha(p) for p in sorted(set(paths))}


def verify_contract(source):
    assert code_hashes() == source['source_code_sha256'], 'Frozen inference code changed'
    for path, digest in source['frozen_files'].items():
        assert sha(path) == digest, path


def jobs(source):
    result = {}
    for model in MODELS:
        for suite in SUITES:
            key = f'{model}-{suite}'
            result[key] = dict(model=model, suite=suite, shard=0, shards=1, phase='suite')
    return result


def task_inputs(source, job, condition):
    original = {r['key']: r for r in source['episodes'] if r['suite'] == job['suite']}
    manifest = read(source['conditions'][condition]['path'])
    pairs = [(original[e['key']], e) for e in manifest['episodes'] if e['key'] in original]
    return pairs[job['shard']::job['shards']]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity(inputs_sha, info, row, entry):
    return dict(inputs_sha256=inputs_sha, model=info['model'], suite=info['suite'], key=row['key'],
        checkpoint_sha256=info['checkpoint_sha256'], row_sha256=digest(row), entry_sha256=digest(entry))


def validate_canary(proof, input_sha, audit):
    assert proof['status']=='passed' and proof['inputs_sha256']==input_sha and proof['gpu_forward_tested']
    assert proof['initialization']==audit
