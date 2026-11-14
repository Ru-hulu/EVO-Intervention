from pathlib import Path
from experiments.soc_disrupted_eval_20261115.common import *

OPT = ROOT/'optimization-20261115'


def runtime_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def backend_hashes():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))
            if p.name in ('backend.py', 'common.py')}


def verify_choice(job_id):
    receipt = read(OPT/'tuning'/f'{job_id}.json')
    assert receipt['status'] == 'passed'
    assert receipt['inputs_sha256'] == sha(ROOT/'inputs.json')
    assert receipt['backend_code_sha256'] == backend_hashes()
    choice = receipt['chosen']
    assert choice['accepted'] and choice['max_abs_error'] <= .002
    assert choice['invalid_positions_equal'] and choice['input_equivalence_verified']
    return receipt
