import copy
from pathlib import Path
from experiments.soc35_finetune_20261114.common import read, rows, save, sha

ROOT = Path('/cache/cvpzhr/setup/recover-spray20-five-finetune-20261116')
OUT = Path('/cache/cvpzhr/outputs/recover-spray20-five-finetune-20261116')
CACHE_SETUP = Path('/cache/cvpzhr/setup/recover-spray20-five-caches-20261116')
CACHE = Path('/cache/cvpzhr/datasets/recover-spray20-five-caches-5fps-20261116')
PACKAGE = 'experiments.recover_spray20_train_20261116'
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
ORDERS = {0: ['robometer-4b', 'forest', 'robobrain25-4b', 'dopamine-3b'], 1: ['procvlm-2b']}
CANDIDATES = {'robometer-4b': [(8, False), (4, False)],
              'procvlm-2b': [(1, False), (2, True)],
              'robobrain25-4b': [(4, False), (2, False)],
              'dopamine-3b': [(4, False), (2, False)],
              'forest': [(4, False), (2, False)]}


def configure(base, micro, gc, pilot=False):
    cfg = copy.deepcopy(base)
    assert cfg['effective_batch'] % (2 * micro) == 0
    cfg.update(microbatch=micro, accumulation=cfg['effective_batch']//(2*micro), gradient_checkpointing=gc)
    if pilot:
        cfg['setup_dir'] = str(ROOT/cfg['name']/'trials'/f'b{micro}-gc{int(gc)}')
    if cfg['name'] == 'forest':
        cfg['native'].update(batch_size=micro, gradient_accumulation_steps=cfg['accumulation'])
        cfg['native']['model']['gradient_checkpointing'] = gc
    assert 2*cfg['microbatch']*cfg['accumulation'] == cfg['effective_batch']
    return cfg


def choose(trials):
    good = [t for t in trials if t['status'] == 'passed' and t['peak_reserved_gib'] <= 63]
    assert good, 'No numerically valid pilot with sufficient GPU reserve'
    return max(good, key=lambda t: t['samples_per_second'])


def verify_code():
    record = read(ROOT/'source-checksums.json')
    for name, digest in record['files'].items():
        assert sha(ROOT/'code'/name) == digest, name


def verify_bindings():
    for name, digest in read(ROOT/'plan.json')['bindings'].items():
        assert sha(name) == digest, name
