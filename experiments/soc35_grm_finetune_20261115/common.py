from pathlib import Path
from experiments.soc35_grm_caches_20261115.common import read, rows, save, sha, MODELS, SUITES
from experiments.soc35_finetune_20261114.common import tensor_digest

ROOT = Path('/cache/cvpzhr/setup/soc35-grm-finetune-20261115')
CACHE = Path('/cache/cvpzhr/datasets/soc35-grm-5fps-20261115')
CACHE_SETUP = Path('/cache/cvpzhr/setup/soc35-grm-caches-20261115')
OUTPUT = Path('/cache/cvpzhr/outputs/soc35-grm-finetune-20261115')
GPUS = {'robobrain25-4b': [0, 1], 'dopamine-3b': [2, 3]}
PACKAGE = 'experiments.soc35_grm_finetune_20261115'


def configure_batch(config, microbatch, checkpointing):
    result = dict(config)
    assert microbatch > 0 and config['effective_batch'] % (2 * microbatch) == 0
    result.update(microbatch=microbatch, accumulation=config['effective_batch'] // (2 * microbatch),
                  gradient_checkpointing=checkpointing)
    return result


def choose_trial(trials):
    good = [r for r in trials if r['status'] == 'passed' and r['peak_reserved_gib'] < 62]
    assert good, 'No throughput pilot passed with sufficient GPU headroom'
    return max(good, key=lambda r: r['samples_per_second'])


def verify_code():
    receipt = read(ROOT/'source-checksums.json')
    for path, expected in receipt['files'].items():
        assert sha(ROOT/'code'/path) == expected, path
