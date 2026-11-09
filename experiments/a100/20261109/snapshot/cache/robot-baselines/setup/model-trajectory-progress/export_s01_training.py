"""Export fixed-rule S01 annotations to ProPriMem and ProgressLite training records."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np

from merge_s01_failure_retry import intervention, link, read, rows, save, WEIGHTS_SHA

TASK = Path('State/01-cover_uncover_blocks')
SUCCESS_SOURCES = ['IMPRINT-Natural-Failures-20261105', 'ProPriMem-IMPRINT-Extra-20261105',
                   'ProPriMem-VLA-Failures-20261104']
CAMERAS = dict(main_view='head_camera', left_view='left_camera', right_view='right_camera')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(states, action_path):
    actions = [a['args'][0] for a in rows(action_path)]
    return hashlib.sha256(np.round(np.asarray(actions), 3).astype('<f4').tobytes()
                          + json.dumps(states[0]['target_state_transition']).encode()).hexdigest()


def success_records(base, excluded):
    seen, selected, duplicates = set(excluded), [], []
    for dataset in SUCCESS_SOURCES:
        root = base / 'viewer-datasets' / dataset
        for row in rows(root / 'metadata.jsonl'):
            if row.get('task_id') != 'S01' or row.get('task_success') is not True \
                    or not row['model_name'].startswith('IMPRINT'):
                continue
            key = row['sample_id']
            pp = root / 'progress' / (key + '.json')
            if not pp.exists() or read(pp).get('weights_file_sha256') != WEIGHTS_SHA:
                continue
            evidence = root / row['evidence_path']
            states_path = evidence if evidence.is_file() else evidence / 'states.jsonl'
            fp = fingerprint(rows(states_path), states_path.parent / 'actions.jsonl')
            identity = (row['model_name'].split(' · ')[0], fp)
            if identity in seen:
                duplicates.append(dict(dataset=dataset, sample_id=key, fingerprint=fp))
                continue
            seen.add(identity)
            prefix = 'imprint-failure-pilot-20261105/fixed-labels' if dataset.startswith('IMPRINT-Natural') \
                else 'imprint-cover-fixed-20261105'
            boundary = base / 'annotations' / prefix / dataset / 'boundaries' / (key + '.json')
            selected.append((root, dict(row, sample_id=dataset + '--' + key, source_sample_id=key,
                source_dataset=dataset, trajectory_source='vla', progress_path=str(pp),
                states_path=str(states_path), actions_path=str(states_path.parent / 'actions.jsonl'),
                subtask_boundary_path=str(boundary), intervention_path=None, action_fingerprint=fp)))
    return selected, duplicates


def check_progress(row, p, boundary, states, delta):
    """Recompute from saved boundaries, independently of the original label builder."""
    n = len(states)
    curve, weights = p['dense_progress'], p['weights']
    assert len(curve) == len(delta) == n == row['total_frames']
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in curve)
    fail = next((f for f, s in enumerate(states) if s['fail_flag']), None)
    success = next((f for f, s in enumerate(states) if s.get('native_success', s.get('success', False))), None)
    assert (success is not None) == row['task_success']
    stop = fail if fail is not None else n
    assert all(curve[f] >= curve[f-1] - 1e-10 for f in range(1, stop))
    expected = [0.] * n
    if row['trajectory_source'] == 'vla':
        base, cursor = 0., 0
        for stage in boundary['normal_stages']:
            a, z = stage['start_frame'], stage['end_frame']
            assert a == cursor and a < z <= n
            excluded = set(stage['excluded_frames'])
            mass = math.fsum(delta[f] for f in range(a, z) if f not in excluded)
            acc = 0.
            for f in range(a, z):
                if f not in excluded:
                    acc += delta[f]
                expected[f] = base + weights[stage['nowid']] * acc / mass
            base += weights[stage['nowid']]
            expected[z-1], cursor = base, z
        expected[cursor:] = [base] * (n-cursor)
        for f in set(boundary['wrong_frames']) | set(boundary['stationary_frames']):
            assert f == 0 or f >= stop or abs(curve[f] - curve[f-1]) < 1e-10
    else:
        cursor = 0
        for stage in boundary['tasks'][0]['episodes'][0]['subtasks']:
            a, z = stage['start_frame'], stage['end_frame']
            assert a == cursor and a < z <= n
            base = math.fsum(weights[i] for i in stage['finishid'])
            mass = math.fsum(delta[a:z]) if stage['scored'] else 1.
            acc = 0.
            for f in range(a, z):
                acc += delta[f]
                expected[f] = base + (weights[stage['nowid']] * acc / mass if stage['scored'] else 0.)
            cursor = z
        assert cursor == n
    if fail is not None:
        expected[fail:] = [0.] * (n-fail)
        assert all(v == 0 for v in curve[fail:])
    if success is not None:
        assert curve[success] == 1 and all(v < 1 for v in curve[:success])
    else:
        assert max(curve) < 1
    assert max(abs(a-b) for a, b in zip(expected, curve)) < 1e-10


def export(base, merged, out):
    task = out / TASK
    for name in ['annotations', 'evidence', 'deltas', *['videos/' + v for v in CAMERAS]]:
        (task / name).mkdir(parents=True, exist_ok=True)
    abnormal = rows(merged / 'supervised.jsonl')
    excluded = {(r['model_name'].split(' · ')[0], r['action_fingerprint'])
                for r in abnormal if r.get('action_fingerprint')}
    positive, duplicates = success_records(base, excluded)
    selected = [(root, row, 'imprint_success') for root, row in positive] \
        + [(merged, row, 'abnormal') for row in abnormal]
    delta_dirs = list((base / 'annotations').glob('**/deltas'))
    rules = merged / 'code/rules'
    weights = read(rules / 'cover_blocks_weights.json')['weights']
    assert sha(rules / 'cover_blocks_weights.json') == WEIGHTS_SHA
    rule_hash = sha(rules / 'cover_blocks_progress.json')
    shutil.copytree(rules, task / 'code/rules')
    labels = out / 'labels' / TASK
    labels.mkdir(parents=True)
    index, records = [], []
    for episode, (root, original, group) in enumerate(selected):
        row = copy.deepcopy(original)
        stem = f'episode_{episode:06d}'
        ev = task / 'evidence' / stem
        ev.mkdir()
        states_path = root / row['states_path']
        states = rows(states_path)
        p = read(root / row['progress_path'])
        bp = root / row['subtask_boundary_path']
        boundary = read(bp)
        assert p['boundary_json_sha256'] == sha(bp)
        assert p['weights'] == weights and p['weights_file_sha256'] == WEIGHTS_SHA
        assert p['rule_sha256'] == rule_hash
        source_root = base / 'viewer-datasets' / row['source_dataset']
        key = row['source_sample_id']
        candidates = [source_root / 'deltas' / (key + '.json')] \
            + [d / (key + '.json') for d in delta_dirs if d.parent.name == row['source_dataset']]
        dp = next(d for d in candidates if d.exists() and sha(d) == p['delta_sha256'])
        delta = read(dp)
        check_progress(row, p, boundary, states, delta['deltas'])
        ann = read(root / row['annotation_path'])
        assert ann['progress'] == p
        views = {}
        for name, camera in CAMERAS.items():
            src = root / row['video_views'][name]['video_path']
            dest = task / 'videos' / name / (stem + '.mp4')
            link(src, dest)
            views[name] = dict(camera=camera, video_path=str(dest.relative_to(task)),
                               video_size_bytes=dest.stat().st_size, video_sha256=sha(dest))
        assert views['main_view']['video_sha256'] == delta['video_sha256']
        assert p.get('video_sha256', delta['video_sha256']) == delta['video_sha256']
        for field, name in [('states_path', 'states.jsonl'), ('actions_path', 'actions.jsonl'),
                            ('subtask_boundary_path', 'boundaries.json'), ('intervention_path', 'intervention.json')]:
            if row.get(field):
                src = root / row[field]
                shutil.copy2(src, ev / name)
                row[field] = str((ev / name).relative_to(task))
        if group == 'imprint_success':
            save(ev / 'intervention.json', intervention(False, None, boundary, states))
            row['intervention_path'] = str((ev / 'intervention.json').relative_to(task))
        for name in ['result.json', 'instruction.json']:
            src = states_path.parent / name
            if src.exists():
                shutil.copy2(src, ev / name)
        shutil.copy2(dp, task / 'deltas' / (stem + '.json'))
        save(ev / 'source_metadata.json', original)
        save(ev / 'source_progress.json', p)
        save(ev / 'source_annotation.json', ann)
        # Training use is requested; retain the original automatic-review qualification explicitly.
        p = dict(p, training_eligible=True, preview_only=False,
                 quality='script_events_verified' if row['trajectory_source'] == 'scripted'
                 else 'automatic_boundaries_not_manually_reviewed')
        ann.update(task_prompt=row['task_prompt'], video_file=views['main_view']['video_path'],
                   fps=row['fps'], total_frames=row['total_frames'], frame_index_base=0,
                   frame_range_convention='[start_frame, end_frame)', progress=p, training_eligible=True)
        cursor = 0
        for i, stage in enumerate(ann['stages']):
            assert stage['start_frame'] == cursor < stage['end_frame'] <= row['total_frames']
            stage['stage_id'] = i
            cursor = stage['end_frame']
        assert cursor == row['total_frames']
        ann['num_stages'] = len(ann['stages'])
        ap = task / 'annotations' / (stem + '.annotations.json')
        save(ap, ann)
        row.update(episode_index=episode, task_name='Cover and uncover blocks', task_slug='cover_uncover_blocks',
                   suite='State', domain='simulation', source_collection='RMBench', has_annotation=True,
                   source_instruction=row.get('source_instruction') or row['task_prompt'],
                   video_path=views['main_view']['video_path'], video_views=views,
                   video_sha256=views['main_view']['video_sha256'], video_size_bytes=views['main_view']['video_size_bytes'],
                   annotation_path=str(ap.relative_to(task)), annotation_sha256=sha(ap), num_stages=ann['num_stages'],
                   progress_path=str((ev / 'progress.json').relative_to(task)), evidence_path=str(ev.relative_to(task)),
                   training_eligible=True, mixture_group=group, annotation_quality=p['quality'])
        # These source-only fields may otherwise retain paths into the original viewer dataset.
        for field in ['joints_path', 'instruction_path']:
            if original.get(field):
                src = root / original[field]
                dest = ev / src.name
                if not dest.exists():
                    link(src, dest)
                row[field] = str(dest.relative_to(task))
        save(ev / 'progress.json', p)
        n, fps = row['total_frames'], row['fps']
        ids = []
        k = 0
        while round(k * fps / 5) < n:
            ids.append(round(k * fps / 5))
            k += 1
        ids = sorted(set([*ids, n-1]))
        record = dict(task=str(TASK), episode=episode, sample_id=row['sample_id'], suite='State', task_id='S01',
            instruction=row['source_instruction'], frames=[dict(index=f, progress=p['dense_progress'][f]) for f in ids],
            dense_progress=p['dense_progress'], progress_source='dense_annotation', progress_label=p['policy'],
            total_frames=n, source_fps=fps, sample_fps=5, source={v['camera']:str(TASK / v['video_path']) for v in views.values()},
            cameras=list(CAMERAS.values()), task_success=row['task_success'], mixture_group=group,
            annotation_path=str(TASK / row['annotation_path']), annotation_quality=p['quality'],
            weights_file_sha256=WEIGHTS_SHA, rule_sha256=rule_hash)
        save(labels / f'episode{episode}.json', record)
        records.append(record)
        index.append(row)
    positive_ids = [r['episode_index'] for r in index if r['mixture_group'] == 'imprint_success']
    negative_ids = [r['episode_index'] for r in index if r['mixture_group'] == 'abnormal']
    # Minimum whole passes over all abnormal records permitting an exact integer 1:2 mix.
    repeats = 2 if len(negative_ids) % 2 else 1
    negative_visits = negative_ids * repeats
    count = len(negative_visits) // 2
    assert count >= len(positive_ids)
    rng = random.Random(20261106)
    whole, rest = divmod(count, len(positive_ids))
    positive_visits = positive_ids * whole + rng.sample(positive_ids, rest)
    rng.shuffle(positive_visits)
    rng.shuffle(negative_visits)
    train = [v for i, p in enumerate(positive_visits) for v in [p, *negative_visits[2*i:2*i+2]]]
    assert len(negative_visits) == 2 * len(positive_visits) and set(train) == set(range(len(index)))
    frequencies = Counter(train)
    for row in index:
        row['train_visits_per_epoch'] = frequencies[row['episode_index']]
    (task / 'metadata.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in index))
    save(task / 'metadata.json', dict(task_id='S01', task_name='Cover and uncover blocks', suite='State',
         task_slug='cover_uncover_blocks', domain='simulation', source_collection='RMBench',
         trajectory_index='metadata.jsonl', trajectory_count=len(index), annotated_trajectory_count=len(index),
         unannotated_trajectory_count=0, video_file_count=3*len(index), annotation_file_count=len(index)))
    save(out / 'labels/_tasks.json', {str(TASK):dict(kind='fixed', suite='State', task_id='S01', rewind_safe=False,
         progress_source='dense_annotation', source_task='cover_blocks')})
    save(out / 'labels/_splits.json', {str(TASK):dict(train=train, all=list(range(len(index))), val=[], pool=[], eval_demo=None)})
    save(out / 'sampling.json', dict(ratio='imprint_success:abnormal=1:2', basis='trajectory visits per epoch',
         unique_success=len(positive_ids), unique_abnormal=len(negative_ids), success_visits=len(positive_visits),
         abnormal_visits=len(negative_visits), seed=20261106, train_episode_order=train))
    save(out / 'excluded.json', dict(duplicate_success_records=duplicates,
         abnormal_source_summary=read(merged / 'summary.json'), missing_or_duplicate_abnormal_excluded=True))
    summary = dict(unique_trajectories=len(index), unique_imprint_success=len(positive_ids), unique_abnormal=len(negative_ids),
         abnormal_breakdown=dict(Counter('scripted_correction' if r['trajectory_source']=='scripted' and r['task_success']
              else 'scripted_failure' if r['trajectory_source']=='scripted' else 'vla_failure'
              for r in abnormal)), training_visits=len(train), success_visits=len(positive_visits),
         abnormal_visits=len(negative_visits), frames=sum(r['total_frames'] for r in index),
         video_files=3*len(index), independently_recomputed_progress=len(index), validation_split='not_assigned',
         manual_boundary_review=False, model_training_started=False)
    save(out / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/cache/robot-baselines'))
    parser.add_argument('--merged', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    with tempfile.TemporaryDirectory(prefix='.s01-training-', dir=args.out.parent) as temp:
        staging = Path(temp) / 'dataset'
        export(args.base, args.merged, staging)
        staging.rename(args.out)
