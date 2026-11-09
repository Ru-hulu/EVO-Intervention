"""Verify clean spray-task cache against all original source-frame PT values."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import prepare_cache as builder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--built', action='store_true')
    args = parser.parse_args()
    started = time.time()
    report = {'mode': 'built-cache-audit' if args.built else 'source-preflight',
              'status': 'checking', 'splits': {}, 'gpu_used': False,
              'training_started': False, 'augmentation_policy': builder.POLICY}
    source_sets = {}
    for split, expected in builder.EXPECTED.items():
        paths = json.loads((builder.SOURCE/f'{split}.sample_files.json').read_text())
        kinds, tasks, source_ids = Counter(), defaultdict(lambda: Counter()), set()
        changed = examples = 0
        for name in paths:
            folder = Path(name)
            summary = json.loads((folder/'summary.json').read_text())
            source = summary['source']
            dense_path = Path(source['progress_csv'])
            assert hashlib.sha256(dense_path.read_bytes()).hexdigest() == source['progress_sha256']
            with dense_path.open() as stream:
                dense = {int(r['frame_index']): float(r['pt']) for r in csv.DictReader(stream)}
            original_path = Path(source['base_dir'])/'samples.jsonl'
            originals = builder.read_rows(original_path)
            forest_rows = builder.read_rows(folder/'samples.jsonl')
            transformed = builder.relabel_replay(forest_rows, originals)
            builder.check_source(summary, transformed, originals, split)
            if args.built:
                rows = builder.read_rows(builder.OUTPUT/'episodes'/split/(summary['episode_id']+'.jsonl'))
                assert len(rows) == len(forest_rows)
            else:
                rows = transformed
            source_ids.add(source['source_id'])
            tasks[source['task']]['episodes'] += 1
            for i, (row, ref) in enumerate(zip(rows, forest_rows, strict=True)):
                frame = int(row.get('source_frame_index', row['frame_index']))
                kind = row.get('kind', 'original')
                assert kind == 'original'
                assert source['task'] == 'inspect-spray-bottles'
                assert abs(row['pt'] - dense[frame]) < 1e-12, (summary['episode_id'], i, 'dense source PT mismatch')
                if kind in ('original', 'stationary'):
                    assert row['pt'] == ref['pt']
                if split == 'validation':
                    assert kind == 'original' and row['pt'] == ref['pt']
                changed += row['pt'] != ref['pt']
                kinds[kind] += 1
                tasks[source['task']]['samples'] += 1
                examples += 1
                if args.built:
                    past = [max(0, j) for j in range(i-7, i)]
                    assert row['history_sample_indices'] == past
                    assert row['window_indices'] == past + [i, i, i]
                    assert row['input_views'] == ['cam_high']*8 + ['cam_left_wrist', 'cam_right_wrist']
                    assert row['missing_current_views'] == []
                    assert len(row['image']) == 10 and row['task'] == summary['task_prompt']
                    assert row['pt_policy'] == builder.POLICY
                    assert row['forest_reference_pt'] == ref['pt']
                    for view, image in zip(builder.VIEWS, row['image'][-3:], strict=True):
                        assert Path(image).samefile(folder/ref['images'][view])
        assert (len(paths), examples) == expected
        assert len(tasks) == builder.TASK_COUNT
        report['splits'][split] = {'episodes': len(paths), 'samples': examples, 'kinds': dict(kinds),
                                  'changed_pt_values': changed, 'tasks': dict(tasks)}
        source_sets[split] = source_ids
        if args.built:
            combined = builder.read_rows(builder.OUTPUT/f'{split}.jsonl')
            assert len(combined) == examples
            expected_ids = set()
            for name in paths:
                summary = json.loads((Path(name)/'summary.json').read_text())
                expected_ids.update(f'{summary["episode_id"]}/{i:08d}' for i in range(summary['num_samples']))
            assert {r['id'] for r in combined} == expected_ids
    assert not source_sets['train'] & source_sets['validation']
    report.update(status='passed', dense_pt_matches_all_observations=True,
                  no_augmentation=True, task='inspect-spray-bottles',
                  original_and_static_labels_unchanged=True, validation_unchanged=True,
                  original_video_splits_disjoint=True, elapsed_seconds=time.time()-started)
    target = builder.SETUP/('label-audit.json' if args.built else 'source-preflight.json')
    builder.write_json(target, report)
    if args.built:
        builder.write_json(builder.OUTPUT/'label-audit.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
