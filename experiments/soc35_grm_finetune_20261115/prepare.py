"""Bind six fresh runs to the audited, unchanged SOC35 native input caches."""
import time
from safetensors import safe_open
from .common import *


def main():
    verify_code()
    assert not (ROOT/'plan.json').exists() and not OUTPUT.exists(), 'Inspect any existing run first'
    audit = read(CACHE/'final-audit.json')
    selection = read(CACHE_SETUP/'selection.json')
    assert audit['status'] == 'passed' and audit['train_validation_disjoint'] and audit['all_tasks_at_least_two']
    assert audit['selection_sha256'] == sha(CACHE_SETUP/'selection.json')
    assert audit['protocol_sha256'] == sha(CACHE_SETUP/'protocol.json')
    configs = {}; files_checked = set()
    for model in MODELS:
        reference = read(f'/cache/cvpzhr/setup/recover-five-finetune-20261114/{model}/config.json')
        for path, h in reference['weights_sha256'].items():
            assert sha(path) == h, path
        configs[model] = {}; cases = []; lengths = {}
        for suite in SUITES:
            folder = CACHE/model/suite
            binding = read(folder/'data-binding.json'); ready = read(folder/'cache-ready.json')
            data = rows(folder/'train.jsonl')
            assert binding['train_sha256'] == ready['train_sha256'] == sha(folder/'train.jsonl')
            assert binding['selection_sha256'] == audit['selection_sha256']
            expected = {e['episode_id'] for e in selection['episodes'] if e['suite'] == suite}
            assert len(expected) == 35 and {r['episode_id'] for r in data} == expected
            assert len(data) == selection['suites'][suite]['observations']
            assert all(r['split'] == 'train' and r['suite'] == suite and r['sample_fps'] == 5 and
                       r['augmentation'] == 'none' and r['uses_future_endpoint_reference'] for r in data)
            for receipt in ready['receipts']:
                for path, h in receipt['files_sha256'].items():
                    if path not in files_checked:
                        assert sha(path) == h, path
                        files_checked.add(path)
            for row in data:
                path = row['prompt_tensor_file']
                if path not in lengths:
                    with safe_open(path, framework='pt', device='cpu') as f:
                        lengths[path] = f.get_slice('input_ids').get_shape()[-1]
                cases.append(row)
            cfg = dict(reference)
            cfg.update(suite=suite, gpus=GPUS[model], train_data=str(folder/'train.jsonl'),
                train_sha256=binding['train_sha256'], selection_sha256=audit['selection_sha256'],
                model_path=binding['model_directory'], official_revision=binding['official_revision'],
                samples=len(data), episodes=35, epochs=1, setup_dir=str(ROOT/model/suite),
                output_dir=str(OUTPUT/model/suite), processor_config={k:binding[k] for k in ('min_pixels','max_pixels')},
                allocator_fraction=.80, min_free_gib=16, num_workers=4, max_length=binding['max_length'],
                pilot_steps=4, training_fps=5, augmentation='none',
                cache_binding_sha256=sha(folder/'data-binding.json'), cache_audit_sha256=sha(CACHE/'final-audit.json'),
                initialization='Independent official base and fresh language LoRA for each suite; no prior adapters')
            assert cfg['effective_batch'] == 16 and cfg['world_size'] == 2
            configs[model][suite] = configure_batch(cfg, 4, False)
        cases.sort(key=lambda r: lengths[r['prompt_tensor_file']])
        for cfg in configs[model].values():
            cfg['pilot_cases'] = [cases[-1], cases[0]]
            cfg['maximum_cached_prompt_tokens'] = lengths[cases[-1]['prompt_tensor_file']]
            cfg['pilot_scope'] = 'Longest and shortest native prompts across all three suites; discard all pilot weights'
            save(Path(cfg['setup_dir'])/'reference-config.json', cfg)
    save(ROOT/'plan.json', dict(configs=configs, suite_order=SUITES, gpus=GPUS,
        epochs=1, effective_batch=16, episodes_per_suite=35, training_fps=5,
        selection_sha256=audit['selection_sha256'], cache_audit_sha256=sha(CACHE/'final-audit.json'),
        previous_adapters_used=False, evaluations_enabled=False, new_automations_created=False))
    save(ROOT/'preflight.json', dict(status='passed', caches=6, rows=38958, artifact_files_verified=len(files_checked),
        all_base_weight_hashes_verified=True, labels_frames_split_unchanged=True, finished=time.time()))


if __name__ == '__main__':
    main()
