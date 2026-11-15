"""Bind audited caches to unchanged per-model Recovery training protocols."""
import time
from .common import *


def main():
    assert not (ROOT/'plan.json').exists(), 'Do not replace a frozen training plan'
    verify_code()
    assert read(CACHE_SETUP/'status.json')['status'] == 'completed'
    assert read(CACHE_SETUP/'final-verification.json')['status'] == 'passed'
    selection = read(CACHE_SETUP/'resolved-selection.json')
    selected = set(selection['selection']['R03'])
    assert len(selected) == 20 and not selected.intersection(selection['excluded_validation'])
    assert all(r['split'] == 'train' and r['task_id'] == 'R03' for r in selection['records'])
    bindings = {}
    def bind(path):
        bindings[str(path)] = sha(path)
        return bindings[str(path)]
    for p in (CACHE_SETUP/'resolved-selection.json', CACHE_SETUP/'final-verification.json',
              CACHE_SETUP/'proc-progress-only-verification.json'):
        bind(p)
    for path, digest in selection['source_sha256'].items():
        assert bind(path) == digest, path
    configs = {}
    for pair, order in ORDERS.items():
        for name in order:
            ready_path = CACHE/name/'cache-ready.json'; ready = read(ready_path); bind(ready_path)
            assert ready['episodes'] == 20 and ready['samples'] == 6002
            base = dict(name=name, suite='Recover-R03', samples=6002, episodes=20, epochs=1, seed=42,
                world_size=2, gpus=[pair*2, pair*2+1], setup_dir=str(ROOT/name), output_dir=str(OUT/name),
                selection_sha256=sha(CACHE_SETUP/'resolved-selection.json'), min_free_gib=16,
                pilot_steps=6, allocator_fraction=.78, fps=5, artificial_augmentation=False)
            if name == 'forest':
                old = Path('/cache/cvpzhr/setup/forest-recover-10-legacy-e1-20261114/training.config.json')
                native = read(old); bind(old)
                assert native['history_source'] == ready['history_source']
                assert {Path(p).name for p in ready['sample_files']} == selected
                native.update(sample_files=ready['sample_files'], data_signature=ready['data_signature'],
                    history_source=ready['history_source'], output_dir=str(OUT/name), epochs=1, max_steps=None,
                    cuda_memory_fraction=.78, save_every_steps=100, save_every_epochs=1,
                    resource_stop_path=str(ROOT/name/'resource-stop.json'))
                native['run_protocol'] = dict(suite='Recover', task='R03', world_size=2, gpus=base['gpus'],
                    effective_batch=8, fps=5, epochs=1, train_episodes=20, train_samples=6002,
                    selection_sha256=base['selection_sha256'], artificial_augmentation=False,
                    initialization='fresh official base, language LoRA, history projection and progress head',
                    old_training_weights_reused=False)
                for p in ready['sample_files']:
                    bind(Path(p)/'samples.jsonl'); bind(Path(p)/'summary.json')
                configs[name] = dict(base, native=native, python=PYTHON, effective_batch=8)
                receipt_path = Path('/cache/cvpzhr/setup/qwen35-original-download-20261113/verification.json')
                receipt = read(receipt_path); bind(receipt_path)
                for filename, item in receipt['files'].items():
                    assert bind(Path(native['model_path'])/filename) == item['sha256'], filename
                continue
            old = (Path('/cache/cvpzhr/setup/procvlm-recover-10-progress-only-e1-20261114/chosen-config.json')
                if name == 'procvlm-2b' else Path('/cache/cvpzhr/setup/recover-five-finetune-20261114')/name/'config.json')
            cfg = read(old); bind(old); cfg.update(base)
            native_path = CACHE_SETUP/f'{name}-native-provenance.json'
            cfg['processor_config'] = read(native_path)['config']; bind(native_path)
            train = CACHE/name/('progress-only/train.jsonl' if name == 'procvlm-2b' else 'train.jsonl')
            records = rows(train)
            assert len(records) == 6002 and {r['episode_id'].rsplit('/', 1)[-1] for r in records} == selected
            assert all(r['split'] == 'train' and r['task_id'] == 'R03' for r in records)
            assert all(r['selection_sha256'] == base['selection_sha256'] for r in records)
            cfg.update(train_data=str(train), train_sha256=bind(train), effective_batch=16)
            for p, digest in cfg['weights_sha256'].items():
                assert bind(p) == digest, p
            configs[name] = cfg
    for name, cfg in configs.items():
        micro = {'robometer-4b':8, 'procvlm-2b':1, 'forest':4, 'robobrain25-4b':4, 'dopamine-3b':4}[name]
        configs[name] = configure(cfg, micro, False)
    save(ROOT/'plan.json', dict(status='prepared', created=time.time(), configs=configs, bindings=bindings,
        pair_orders=ORDERS, epochs=1, samples_per_model=6002, episodes=20, fresh_official_initialization=True,
        pilot_policy='No pilots: user requested immediate training using previously successful Recovery configurations',
        evaluation_enabled=False, desktop_automation_created=False))
    print('Prepared five models: 20 identical R03 train episodes, 6002 samples/model, no validation leakage', flush=True)


if __name__ == '__main__': main()
