from pathlib import Path
from experiments.soc35_finetune_20261114.common import read, rows, save, sha, tensor_digest

ROOT = Path('/cache/cvpzhr/setup/recover-spray20-single-validation-20261116')
OUT = Path('/cache/cvpzhr/evaluations/recover-spray20-single-validation-20261116')
IMAGES = ROOT/'images'
TRAIN = Path('/cache/cvpzhr/setup/recover-spray20-five-finetune-20261116')
TRAIN_SELECTION = Path('/cache/cvpzhr/setup/recover-spray20-five-caches-20261116/resolved-selection.json')
CACHE_SETUP = TRAIN_SELECTION.parent
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
REPORT_PYTHON = '/cache/cvpzhr/envs/robometer/bin/python'
PACKAGE = 'experiments.recover_spray20_single_20261116'
MODELS = ['procvlm-2b', 'forest', 'robometer-4b', 'robobrain25-4b', 'dopamine-3b']


def result_path(name, key): return OUT/name/'episodes'/(key.replace('/', '__')+'.json')


def frame_inputs(row):
    return dict(key=row['key'], task=row['task_prompt'], frames=[dict(frame_index=i,
        image=[str(IMAGES/row['task_id']/row['sample_id']/f'{i:08d}-{v}.png') for v in range(3)])
        for i in row['frame_indices']])


def verify_code():
    for path, digest in read(ROOT/'source-checksums.json')['files'].items():
        assert sha(ROOT/'code'/path) == digest, path


def validate_proof(name, proof, config):
    assert proof['status'] == 'passed' and not proof['pilot'] and proof['epoch'] == 1
    assert proof['expected_episodes'] == 20 and proof['expected_samples'] == 6002
    assert proof['selection_sha256'] == sha(TRAIN_SELECTION) == config['selection_sha256']
    assert config['name'] == name and config['suite'] == 'Recover-R03'
    assert proof['fresh_official_initialization']
    assert proof['checkpoint'].startswith(config['output_dir']+'/')
    if name == 'forest':
        assert proof['optimizer_constant_lr_two_rank_rng'] and proof['steps'] == 751
    else:
        assert proof['optimizer_scheduler_two_rank_rng'] and proof['replicas_equal'] and proof['steps'] == 376
        assert proof['train_input_sha256'] == sha(config['train_data'])
