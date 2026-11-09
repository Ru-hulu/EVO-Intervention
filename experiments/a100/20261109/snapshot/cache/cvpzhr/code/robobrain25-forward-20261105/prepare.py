"""Derive forward GRM labels and reuse source images/pixels, without changing PT."""
from collections import Counter
from pathlib import Path
import hashlib
import json
from grm import answer

ROOT = Path('/cache/cvpzhr')
OUTPUT = ROOT / 'datasets/robobrain25-forward-cap50-static-jitter-20261105'
FOREST = ROOT / 'datasets/forest-proprimem-cap50-5fps-static-jitter-seed42-20261104'
ROBOMETER = ROOT / 'datasets/robometer-proprimem-cap50-5fps-h7-current3-static-jitter-seed42-20261104'
AUGVAL = ROOT / 'setup/forest-robometer-augval-pearson-20261105/plan.json'
VIEWS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    OUTPUT.mkdir(exist_ok=True)
    assert not (OUTPUT / 'manifest.json').exists(), 'Completed dataset already exists; do not overwrite'
    train_plan, val_plan, rb = read(FOREST / 'index.json'), read(AUGVAL), read(ROBOMETER / 'manifest.json')
    rb_episodes = {e['episode_id']: e for split in rb['splits'].values() for e in split['episodes']}
    train_ids = {e['episode_id'] for e in train_plan['episodes']}
    val_ids = {e['episode_id'] for e in val_plan['episodes']}
    assert len(train_ids) == 1656 and len(val_ids) == 417 and train_ids.isdisjoint(val_ids)
    episodes, report, protected = {}, {}, {}
    quantization_max = 0.0
    for split, entries in [('train', train_plan['episodes']), ('validation', val_plan['episodes']),
                           ('augmented_validation', val_plan['episodes'])]:
        counts, kinds, n = Counter(), Counter(), 0
        with (OUTPUT / f'{split}.jsonl').open('w') as output:
            for e in entries:
                source = Path(e['source_dir']); eid = e['episode_id']
                original = rows(source / 'samples.jsonl'); summary = read(source / 'summary.json')
                for name in ['samples.jsonl', 'summary.json']:
                    protected[str(source / name)] = sha(source / name)
                assert original[0]['frame_index'] == 0 and original[0]['pt'] == 0
                assert original[-1]['frame_index'] == summary['source']['frames'] - 1 and original[-1]['pt'] == 1
                timeline_path = Path(e['output_dir']) / 'samples.jsonl' if split == 'train' else (
                    Path(e['timeline']) if split == 'augmented_validation' else source / 'samples.jsonl')
                timeline = rows(timeline_path)
                protected[str(timeline_path)] = sha(timeline_path)
                if split == 'train':
                    end_pixel = next(i for i, r in enumerate(timeline)
                        if r['kind'] == 'original' and r['source_sample_index'] == len(original) - 1)
                else:
                    end_pixel = len(original) - 1
                descriptor = dict(source_dir=str(source), task_prompt=summary['task_prompt'],
                    suite=eid.split('/')[0], task_id='/'.join(eid.split('/')[:-1]),
                    native_images=[[r['images'].get(v) for v in VIEWS] for r in original],
                    pixel_cache=rb_episodes[eid]['pixel_cache'], end_pixel_index=end_pixel,
                    endpoint_native_frames=[0, original[-1]['frame_index']],
                    native_samples=len(original), same_episode_final_frame_reference=True)
                assert Path(descriptor['pixel_cache']).is_file()
                if eid in episodes:assert episodes[eid] == descriptor
                episodes[eid] = descriptor
                for r in original:
                    for name in r['images'].values():
                        if name:assert (source / name).is_file(), source / name
                for i, r in enumerate(timeline):
                    src = r.get('source_sample_index', i)
                    anchor = r.get('anchor_sample_index', src)
                    assert r['images'] == original[src]['images']
                    assert r['pt'] == original[anchor]['pt']
                    row = dict(episode_id=eid, sample_index=i, source_sample_index=src,
                        anchor_sample_index=anchor, frame_index=r['frame_index'],
                        source_frame_index=original[src]['frame_index'],
                        timestamp_seconds=r.get('timestamp_seconds', r['frame_index'] / summary['source']['fps']),
                        kind=r.get('kind', 'original'), pt=r['pt'], answer=answer(r['pt']),
                        pixel_index=i if split == 'train' else src)
                    quantization_max = max(quantization_max, abs(float(row['answer'][7:-9]) / 100 - r['pt']))
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    n += 1; counts[descriptor['suite']] += 1; kinds[row['kind']] += 1
        report[split] = dict(samples=n, episodes=len(entries), by_suite=dict(counts), by_kind=dict(kinds))
    assert [report[s]['samples'] for s in report] == [318931, 66891, 78554]
    assert len(episodes) == 2073
    write(OUTPUT / 'episodes.json', episodes)
    write(OUTPUT / 'source-hashes.json', protected)
    write(OUTPUT / 'manifest.json', dict(status='completed', splits=report,
        input_contract='grm-forward-same-episode-goal-8images-v1',
        image_order=['reference_start_main', 'reference_end_main', 'before_main', 'before_left',
                     'before_right', 'after_main', 'after_left', 'after_right'],
        label='100 * existing PT, one decimal percent; native PT retained for evaluation',
        training_augmentation=train_plan['contract'], augmented_validation_plan_sha256=val_plan['plan_sha256'],
        source_training_plan_sha256=train_plan['plan_sha256'], uses_future_goal_reference=True,
        endpoint_semantics='User-selected episode final frame; PT=1 checked, semantic success not independently reannotated',
        missing_view_policy='RGB black 64x64 for entire absent camera; partial missing episode remains excluded',
        excluded_episode='Order/03-swap_blocks_with_confirmation/episode_001342',
        maximum_label_quantization_error_pt=quantization_max,
        forest_memory_features_used=False, existing_pixel_cache_reused=True,
        source_PT_unchanged=True, training_started=False))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
