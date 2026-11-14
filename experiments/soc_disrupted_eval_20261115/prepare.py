"""Explicit CPU-only preparation; never launches inference or modifies source data."""
import argparse
from collections import Counter
from .common import *
from .protocol import timeline
from experiments.prm_five_groups_20261114.stream_images import verify_cache
from experiments.prm_five_groups_20261114.stream_manifests import freeze


def build_registry(train):
    registry = {}; files = {str(TRAIN_SELECTION):sha(TRAIN_SELECTION)}
    for suite in SUITES:
        ref = train['suites'][suite]; proof = read(ref['checkpoint_verification_source'])
        assert proof['status'] == 'completed' and proof['final_checkpoint_verified'] and proof['epochs'] == 1
        checkpoint = ref['forest_checkpoint']; h = sha(checkpoint)
        assert h == ref['forest_checkpoint_sha256'] == proof['checkpoint_sha256']
        config = read(ref['forest_config'])
        assert config['run_protocol']['suite'] == suite and config['run_protocol']['train_episodes'] == 35
        assert config['event_split_method'] == 'max_depth'
        registry[f'forest-{suite}'] = dict(model='forest', suite=suite, python=PYTHON,
            checkpoint=checkpoint, checkpoint_sha256=h, config=config,
            proof=ref['checkpoint_verification_source'], proof_sha256=sha(ref['checkpoint_verification_source']))
        for model in ('procvlm-2b', 'robometer-4b'):
            p = TRAIN/model/suite/'formal/verification.json'; proof = read(p)
            assert proof['status'] == 'passed' and not proof['pilot'] and proof['epoch'] == 1
            assert proof['expected_episodes'] == 35 and proof['selection_sha256'] == sha(TRAIN_SELECTION)
            ck = Path(proof['checkpoint']); cfg = read(ck/'run_config.json')
            assert cfg['suite'] == suite and cfg['episodes'] == 35 and cfg['epochs'] == 1
            assert cfg['selection_sha256'] == sha(TRAIN_SELECTION)
            assert sha(ck/'trainable-state.pt') == proof['trainable_state_sha256']
            registry[f'{model}-{suite}'] = dict(model=model, suite=suite, python=cfg['python'], config=cfg,
                checkpoint=str(ck), checkpoint_sha256=proof['trainable_state_sha256'],
                proof=str(p), proof_sha256=sha(p))
            for path,digest in cfg['weights_sha256'].items():
                files[path] = digest
            files[str(ck/'trainable-state.pt')] = proof['trainable_state_sha256']
            files[str(ck/'run_config.json')] = sha(ck/'run_config.json')
            files[str(ck/'trainable-state.json')] = sha(ck/'trainable-state.json')
        files[checkpoint] = h
        files[ref['forest_config']] = ref['forest_config_sha256']
        for model in MODELS:
            info = registry[f'{model}-{suite}']; files[info['proof']] = info['proof_sha256']
    # Freeze original base/tokenizer/native code provenance as well as adapters.
    base = read('/cache/cvpzhr/setup/qwen35-original-download-20261113/verification.json')
    assert base['all_official_hashes_verified']
    for name,item in base['files'].items():
        files[str(Path(base['destination'])/name)] = item['sha256']
    for model in ('procvlm-2b','robometer-4b'):
        p = Path('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114')/f'{model}-native-provenance.json'
        if not p.exists():
            p = Path('/cache/cvpzhr/setup/recover-six-model-caches-20261114')/f'{model}-native-provenance.json'
        native = read(p); files[str(p)] = sha(p)
        model_root = registry[f'{model}-State']['config']['model_path']
        for path, h in native['source_sha256'].items():
            if Path(path).is_relative_to(model_root): files[path] = h
    # The SOC trainer imports this exact native Robometer runtime, rather than
    # the old zero-shot snapshot cited by the model download receipt.
    for folder in ('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114/runtime',
                   '/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream'):
        paths = list(Path(folder).rglob('*.py')); assert paths
        files.update({str(p):sha(p) for p in paths})
    clip = Path(registry['forest-State']['config']['history_source']['memory_config']['clip_model_path'])
    assert clip.exists()
    files.update({str(p):sha(p) for p in clip.rglob('*') if p.is_file() and '.cache' not in p.parts})
    return registry, files


def prepare():
    verification = read(DATA/'verification.json')
    assert verification['status'] == 'ready' and verification['all_playback_points_verified']
    preparation = read(DATA/'preparation.json'); sources = read(DATA/'inputs.json')
    assert sha(DATA/'inputs.json') == preparation['inputs_sha256']
    selection = read(DATA_SETUP/'selection.json'); train = read(TRAIN_SELECTION)
    assert sources['selection_sha256'] == sha(DATA_SETUP/'selection.json')
    rows = sources['episodes']
    assert len(rows) == 84 and not {r['key'] for r in rows} & {r['episode_id'] for r in train['episodes']}
    registry, files = build_registry(train)
    files.update(selection['source_hashes'])
    for p in (DATA/'verification.json', DATA/'preparation.json', DATA/'inputs.json', DATA_SETUP/'selection.json'):
        files[str(p)] = sha(p)
    conditions = {}
    for row in rows:
        assert row['split'] == 'val' and not row['input_issues']
        files[row['progress_path']] = row['progress_sha256']
        cache = Path(row['image_root'])
        verify_cache(cache, row, dict(video_hashes=row['video_hashes'], frame_indices=row['frame_indices']), False)
        files[str(cache/'verification.json')] = sha(cache/'verification.json')
    for suite in SUITES:
        assert sum(r['suite'] == suite for r in train['episodes']) == 35
        counts = Counter(r['task_id'] for r in rows if r['suite'] == suite)
        assert set(counts.values()) == {2}
    by_key = {r['key']:r for r in rows}
    for condition in CONDITIONS:
        p = DATA/'manifests'/f'{condition}.json'
        assert sha(p) == preparation['manifests'][str(p)]
        d = read(p); assert len(d['episodes']) == 84 and {e['key'] for e in d['episodes']} == set(by_key)
        for entry in d['episodes']:
            timeline(by_key[entry['key']], entry)
            assert sha(entry['playback_csv']) == entry['playback_csv_sha256']
            files[entry['playback_csv']] = entry['playback_csv_sha256']
        files[str(p)] = sha(p)
        conditions[condition] = dict(path=str(p), sha256=sha(p), episodes=84, observations=d['expected_samples'])
    protocol = dict(models=list(MODELS), weights='matching suite SOC35 final 1 epoch', memory_enabled=True,
        input_fps=5, seeds=list(SEEDS), batch=1, future_endpoint_reference=False,
        procvlm='recent8 images; progress-only prompt; no textual memory',
        robometer='native uniform-prefix 10-image interface', forest='native CLIP/max_depth, fixed capacity',
        circle_consistency='mean of (abs(p3-p1)+abs(p2-p1)+abs(p3-p2))/3 over all event-local interior triplets per trajectory',
        loop_turning_endpoints='excluded: only two visits', overlapping_loops='pair separately per insertion against original forward visit',
        static_mae='mean abs(pred-segment_mean) within inserted frames only; two segment MAEs equally averaged per trajectory',
        aggregation='equal trajectories within task/suite; seeds reported separately and mean/sample SD across three seeds',
        invalid_predictions='raw/null retained; any invalid required value makes the event/trajectory metric undefined; no subset substitution',
        normalization='unit progress; no clipping/smoothing/per-curve rescaling')
    value = dict(schema='soc35-disrupted-memory-v1', episodes=rows, registry=registry, conditions=conditions,
        source_code_sha256=code_hashes(), frozen_files=files, protocol=protocol,
        expected_curves=3*504, expected_observations=3*sum(v['observations'] for v in conditions.values()),
        evaluation_started=False)
    verify_contract(value)
    freeze(ROOT/'inputs.json', value)
    freeze(ROOT/'preflight.json', dict(status='passed', registry_entries=9, episodes=84,
        curves=value['expected_curves'], observations=value['expected_observations'],
        inputs_sha256=sha(ROOT/'inputs.json'), gpu_forward_tested=False, evaluation_started=False))
    if not (ROOT/'status.json').exists():
        save(ROOT/'status.json',dict(status='prepared_not_started',inputs_sha256=sha(ROOT/'inputs.json'),
            evaluation_started=False,gpu_forward_tested=False,registry_entries=9))
    return dict(status='prepared_not_started', curves=value['expected_curves'], observations=value['expected_observations'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only', action='store_true', required=True)
    parser.parse_args();print(json.dumps(prepare()))
