from pathlib import Path
from experiments.recover_finetune_20261114.common import read, rows, save, sha

ROOT = Path('/cache/cvpzhr/setup/forest-spray20-historydrop50-e2-20261116')
OUT = Path('/cache/cvpzhr/outputs/forest-spray20-historydrop50-e2-20261116')
OLD = Path('/cache/cvpzhr/setup/recover-spray20-five-finetune-20261116')
CACHE_SETUP = Path('/cache/cvpzhr/setup/recover-spray20-five-caches-20261116')
CACHE = Path('/cache/cvpzhr/datasets/recover-spray20-five-caches-5fps-20261116/forest')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
PACKAGE = 'experiments.forest_spray20_historydrop_20261116'


def verify_code():
    for name, digest in read(ROOT/'source-checksums.json')['files'].items():
        assert sha(ROOT/'code'/name) == digest, name


def verify_bindings():
    for path, digest in read(ROOT/'plan.json')['bindings'].items():
        assert sha(path) == digest, path
