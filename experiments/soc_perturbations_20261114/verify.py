"""Independent full-cache checks. Never trust a generator's success flag alone."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import time

from .prepare import ROOT, CACHE, SUITES, SEEDS, VISION_SHA, EXCLUDED, read, choose
from experiments.prm_five_groups_20261114.prepare_cache import save, sha, progress_values
from experiments.prm_five_groups_20261114.stream_images import verify_cache
from experiments.prm_five_groups_20261114.stream_runtime import validate_entry


def verify_playback(row, entry):
    """Reconstruct from the two source anchors without calling playback_map."""
    validate_entry(row, entry)
    assert entry['seed'] in SEEDS and entry['memory_enabled'] is True
    assert entry['condition'] in ('static_only', 'rewind_only')
    assert len(entry['events']) == 2
    anchors = [e['after_source_step'] for e in entry['events']]
    assert len(set(anchors)) == 2 and anchors == sorted(anchors)
    expected, times, inserted = [], [], 0
    events = {e['after_source_step']: e for e in entry['events']}
    for i, t in enumerate(row['timestamps']):
        expected.append(i); times.append(t+inserted/5)
        if i not in events:
            continue
        event = events[i]; length = event['duration_steps']
        assert type(length) is int and 5 <= length <= 15
        assert 15 <= i < len(row['frame_indices'])-1
        assert event['first_inserted_step'] == len(expected)
        if entry['condition'] == 'static_only':
            path = [i for _ in range(length)]
        else:
            path = [i-j for j in range(1, length+1)] + [i-length+j for j in range(1, length+1)]
        expected += path
        anchor_t = times[-1]
        times += [anchor_t+j/5 for j in range(1, len(path)+1)]
        inserted += len(path)
        assert event['inserted_steps'] == len(path) and event['last_inserted_step'] == len(expected)-1
        assert event['one_way_duration_seconds'] == length/5
    assert expected == entry['source_step_indices']
    assert times == entry['playback_timestamps']
    assert entry['source_timestamps'] == [row['timestamps'][i] for i in expected]
    assert entry['source_frame_indices'] == [row['frame_indices'][i] for i in expected]
    assert entry['target_pt'] == [row['target_pt'][i] for i in expected]
    assert entry['progress_sha256'] == row['progress_sha256']


def verify():
    import numpy as np
    selection = read(ROOT/'selection.json'); preparation = read(CACHE/'preparation.json')
    assert sha(CACHE/'inputs.json') == preparation['inputs_sha256']
    baseline = read(CACHE/'inputs.json')
    assert baseline['selection_sha256'] == sha(ROOT/'selection.json')
    rows = baseline['episodes']; by_key = {r['key']: r for r in rows}
    assert len(rows) == len(by_key) == 84 and EXCLUDED not in by_key
    tasks = {s: [t for t in selection['tasks'] if t['suite'] == s] for s in SUITES}
    expected = [f'{s}/{t}/{k}' for s, t, k in choose(tasks)]
    assert list(by_key) == expected
    assert Counter((r['suite'], r['task_id']) for r in rows) == {
        (t['suite'], t['task_id']): 2 for t in selection['tasks']}
    for path, digest in selection['source_hashes'].items():
        assert sha(path) == digest, f'Original data changed: {path}'
    sources = {r['key']: r for r in selection['episodes']}
    label_frames, pngs, absent_views = 0, 0, 0
    for row in rows:
        original = sources[row['key']]
        for k in original:
            if k != 'progress_sha256':
                assert row[k] == original[k], (row['key'], k)
        values = progress_values(row)
        provenance = read(Path(row['progress_path']).with_suffix('.json'))
        assert provenance['policy'] == 'whole-video-cumulative-visual-delta-v1'
        assert provenance['vision_sha256'] == VISION_SHA
        assert provenance['model_revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
        assert provenance['video_sha256'] == row['video_hashes'][0]
        assert provenance['csv_sha256'] == row['progress_sha256']
        assert provenance['all_original_frames'] and not provenance['annotations_used']
        native = list(csv.DictReader(Path(row['progress_path']).open()))
        delta = np.array([float(v['delta']) for v in native], dtype=np.float64)
        assert delta[0] == 0 and np.isfinite(delta).all() and (delta >= 0).all()
        cumulative = np.cumsum(delta, dtype=np.float64)
        assert cumulative[-1] > 1e-8
        assert np.array_equal(cumulative/cumulative[-1], np.array(list(values.values())))
        assert row['target_pt'] == [values[i] for i in row['frame_indices']]
        assert list(values)[0] == 0 and values[0] == 0 and values[row['frames']-1] == 1
        root = Path(row['image_root'])
        verify_cache(root, row, dict(video_hashes=row['video_hashes'], frame_indices=row['frame_indices']), False)
        pngs += len(read(root/'verification.json')['png_sha256'])
        absent_views += row['videos'].count(None)
        label_frames += len(native)
    versions = observations = 0
    conditions = {}
    for path, h in preparation['manifests'].items():
        assert sha(path) == h
        manifest = read(path)
        assert manifest['baseline_manifest_sha256'] == sha(CACHE/'inputs.json')
        entries = manifest['episodes']
        assert len(entries) == len({e['key'] for e in entries}) == 84
        assert {e['key'] for e in entries} == set(by_key)
        count = 0
        for entry in entries:
            row = by_key[entry['key']]
            verify_playback(row, entry)
            assert f"{entry['condition']}-seed{entry['seed']}" == manifest['condition']
            assert sha(entry['playback_csv']) == entry['playback_csv_sha256']
            records = list(csv.DictReader(Path(entry['playback_csv']).open()))
            assert len(records) == len(entry['target_pt'])
            for i, item in enumerate(records):
                assert int(item['step']) == i
                frame = int(item['source_frame_index'])
                assert frame == entry['source_frame_indices'][i]
                assert float(item['pt']) == entry['target_pt'][i]
                assert float(item['playback_seconds']) == entry['playback_timestamps'][i]
                assert float(item['source_seconds']) == entry['source_timestamps'][i]
                for v, key in enumerate(('main_image', 'left_image', 'right_image')):
                    expected_path = str(Path(row['image_root'])/f'{frame:08d}-{v}.png') if row['videos'][v] else ''
                    assert item[key] == expected_path
                    if expected_path:
                        assert Path(expected_path).exists()
            count += len(records)
        assert count == manifest['expected_samples']
        conditions[manifest['condition']] = dict(versions=len(entries), observations=count)
        versions += len(entries); observations += count
    assert set(conditions) == {f'{v}-seed{s}' for v in ('static_only', 'rewind_only') for s in SEEDS}
    assert versions == 504
    receipt = dict(status='ready', checked_at=time.time(), tasks=42, episodes=84, versions=versions,
        by_suite=dict(Counter(r['suite'] for r in rows)), observations=observations, conditions=conditions,
        all_playback_points_verified=True, native_label_frames=label_frames, unique_png_files=pngs,
        missing_original_views_using_existing_policy=absent_views,
        original_files_unchanged=True, train_val_split_unchanged=True,
        supplemental_visual_pt=len(selection['missing_pt']), evaluation_started=False,
        source_selection_sha256=sha(ROOT/'selection.json'), preparation_sha256=sha(CACHE/'preparation.json'))
    save(CACHE/'verification.json', receipt)
    save(ROOT/'status.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False))
