"""Record runtime readiness separately from dataset/evaluation readiness."""
import time
from pathlib import Path

from common import MODELS, ROOT, read, sha, write

RESTORE = Path('/cache/cvpzhr/setup/zeroshot-rebuild-20261113')
ENV = RESTORE / 'environment'


def main():
    weights = read(RESTORE / 'weights/restoration-verification.json')
    assert weights['status'] == 'verified' and weights['all_model_files_hash_verified']
    assert set(weights['models']) == set(MODELS)
    assert not weights['model_bytes_transit_local_computer']
    processor = read(RESTORE / 'weights/processor-verification.json')
    assert processor['all_pinned_hf_hashes_match']
    installed = read(ENV / 'installation-status.json')
    assert installed['status'] == 'installed'
    environments = {}
    for group in ('robometer', 'generative', 'vlac'):
        state = read(ENV / f'{group}-status.json')
        lock = ENV / f'{group}-environment.lock.txt'
        assert state['status'] == 'installed'
        assert sha(lock) == state['lock_sha256']
        assert sha(ROOT / f'requirements-{group}.txt') == state['requirements_sha256']
        environments[group] = {k: state[k] for k in ('environment', 'requirements_sha256', 'lock_sha256')}
    provenance = read(ROOT / 'model-provenance.json')
    models = {}
    for name, config in MODELS.items():
        state = read(ENV / 'smoke' / f'{name}.json')
        assert state['status'] == 'passed' and state['native_forward_passed']
        assert not state['local_adapter_loaded'] and not state['dataset_evaluation_started']
        assert state['worker_sha256'] == sha(ROOT / 'worker.py')
        assert state['official_revision'] == provenance[name]['revision'] == weights['models'][name]['revision']
        assert state['checkpoint'] == config['path'] == weights['models'][name]['path']
        assert state['config_sha256'] == sha(Path(config['path']) / 'config.json')
        models[name] = {k: state[k] for k in ('checkpoint', 'official_revision', 'versions',
                         'native_forward_passed', 'worker_sha256', 'config_sha256')}
    queue = read(ENV / 'smoke/status.json')
    assert queue['status'] == 'passed' and set(queue['passed']) == set(MODELS)
    assert not queue['failed'] and not queue['active'] and not queue['pending']
    result = {
        'runtime_environment_ready': True,
        'dataset_preflight_passed': False,
        'dataset_evaluation_started': False,
        'notify_user_before_dataset_evaluation': True,
        'models_passed': len(models), 'models': models, 'environments': environments,
        'model_files_hash_verified': sum(m['verified_files'] for m in weights['models'].values()),
        'processor_files_hash_verified': len(processor['files']),
        'model_bytes_transit_local_computer': False,
        'dataset_access_check': read(ENV / 'dataset-access-check.json'),
        'dataset_preflight_pending_reason': 'Validation samples, original PT labels and exact split not yet restored.',
        'code_restoration': read(RESTORE / 'code-restoration.json'),
        'verified_at': time.time(),
    }
    write(ENV / 'environment-readiness.json', result)
    print(f"Runtime checks passed: {len(models)}/9. Dataset evaluation has NOT started.")


if __name__ == '__main__':
    main()
