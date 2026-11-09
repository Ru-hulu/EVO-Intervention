"""Build a standalone S01 collection from existing failure and correction records."""
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from collections import Counter

import numpy as np

BASE = Path('/cache/robot-baselines')
VIEWERS = BASE / 'viewer-datasets'
SCRIPTED = 'RMBench-CoverBlocks-40-20261105'
PILOT = 'IMPRINT-Natural-Failures-20261105'
SOURCES = ['ProPriMem-VLA-Failures-20261104', 'ProPriMem-IMPRINT-Extra-20261105',
           PILOT, 'ProPriMem-RenderAB-20261104', SCRIPTED]
WEIGHTS_SHA = '51b08b95efe7f1c561f076655a8a8055e688803de4adc5216002d1b19a7a75d8'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def link(source, dest):
    try:
        os.link(source, dest)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, dest)


def intervention(scripted, result, boundary, states):
    if scripted:
        events = {e['kind']: e['frame'] for e in result['events']}
        error_event = next(e for e in result['events'] if e['kind'] == 'memory_error')
        start = events['memory_error']
        hover = events['wrong_hover']
        correction = events.get('memory_retrieval')
        uncovered = events.get('first_uncovered')
        end = correction if correction is not None else uncovered
        wrong_cover = error_event['corrupted']
        assert start <= hover < end <= len(states)
        assert all(s['covered'][wrong_cover] and not s['fail_flag'] for s in states[start:end])
        return dict(source='script_events_verified_against_cover_state', frame_range='[start,end)',
                    pre_lift_window=[start, end], wrong_hover_frame=hover,
                    preferred_hover_window=[hover, events.get('wrong_open_start', end)],
                    correction_start_frame=correction, correct_target_reached_frame=events.get('restored'),
                    wrong_open_start_frame=events.get('wrong_open_start'), first_wrong_uncovered_frame=uncovered,
                    native_failure_frame=events.get('irreversible_failure'), wrong_cover=wrong_cover,
                    correct_cover=error_event['truth'],
                    history_reference_frame=result['memory']['source_frame'])
    if not boundary:
        return None
    uncovered = boundary.get('first_wrong_uncovered_frame')
    window = None
    if uncovered is not None and 0 < uncovered < len(states):
        phase = states[uncovered]['current_state_pointer']
        if phase >= 3:
            candidates = [f for f in boundary['wrong_frames'] if 0 <= f < uncovered
                          and states[f]['current_state_pointer'] == phase and not states[f]['fail_flag']]
            if candidates:
                last = candidates[-1]
                first = last
                for f in reversed(candidates[:-1]):
                    if f != first - 1:
                        break
                    first = f
                window = [first, last + 1]
    return dict(source='automatic_motion_and_geometry_needs_review', frame_range='[start,end)',
                pre_lift_window=window, first_wrong_lift_frame=boundary.get('first_wrong_lift_frame'),
                first_wrong_uncovered_frame=uncovered, native_failure_frame=boundary.get('native_failure_frame'),
                wrong_cover=boundary.get('wrong_cover'))


def merge(output):
    all_rows, labeled, fingerprints = [], [], {}
    for dataset in SOURCES:
        source = VIEWERS / dataset
        scripted = dataset == SCRIPTED
        for original in rows(source / 'metadata.jsonl'):
            if scripted:
                if original.get('trajectory_class') not in ('Failure', 'MemoryCorrection', 'Recover'):
                    continue
            elif original.get('task_id') != 'S01' or original.get('task_success') is not False:
                continue
            key = original['sample_id']
            sample_id = f'{dataset}--{key}'
            folder = output / 'episodes' / sample_id
            folder.mkdir(parents=True)
            evidence = source / original['evidence_path']
            states_path = evidence if evidence.is_file() else evidence / 'states.jsonl'
            states = rows(states_path)
            n = original['total_frames']
            assert len(states) == n, sample_id
            link(states_path, folder / 'states.jsonl')
            relative = folder.relative_to(output)
            row = dict(original, sample_id=sample_id, task_id='S01', source_dataset=dataset,
                       source_sample_id=key, trajectory_source='scripted' if scripted else 'vla',
                       evidence_path=str(relative), states_path=str(relative / 'states.jsonl'),
                       training_eligible=False)
            row['video_views'] = {}
            for name, view in original['video_views'].items():
                src = source / view['video_path']
                dest = folder / f'{name}.mp4'
                link(src, dest)
                row['video_views'][name] = dict(view, video_path=str(dest.relative_to(output)))
            row['video_path'] = row['video_views']['main_view']['video_path']
            source_ann = original.get('annotation_path')
            row['annotation_path'] = None
            if source_ann:
                shutil.copy2(source / source_ann, folder / 'annotation.json')
                row['annotation_path'] = str(relative / 'annotation.json')
            action_path = states_path.parent / 'actions.jsonl'
            row['duplicate_of'] = None
            if action_path.exists():
                link(action_path, folder / 'actions.jsonl')
                row['actions_path'] = str(relative / 'actions.jsonl')
                actions = [a['args'][0] for a in rows(action_path)]
                signature = hashlib.sha256(np.round(np.asarray(actions), 3).astype('<f4').tobytes()
                    + json.dumps(states[0]['target_state_transition']).encode()).hexdigest()
                row['action_fingerprint'] = signature
                fingerprint_key = (original['model_name'].split(' · ')[0], signature)
                row['duplicate_of'] = fingerprints.get(fingerprint_key)
                fingerprints.setdefault(fingerprint_key, sample_id)
            progress_path = source / 'progress' / (key + '.json')
            row['progress_path'] = None
            if progress_path.exists():
                progress = read(progress_path)
                values = progress['dense_progress']
                assert len(values) == n and all(math.isfinite(v) and -1e-8 <= v <= 1+1e-8 for v in values), sample_id
                assert progress['weights_file_sha256'] == WEIGHTS_SHA, sample_id
                if original['task_success']:
                    assert abs(values[-1] - 1) < 1e-8, sample_id
                shutil.copy2(progress_path, folder / 'progress.json')
                row['progress_path'] = str(relative / 'progress.json')
            if scripted:
                result = read(states_path.parent / 'result.json')
                for field, name in [('instruction_path', 'instruction.json'), ('subtask_boundary_path', 'boundaries.json')]:
                    if original.get(field):
                        shutil.copy2(source / original[field], folder / name)
                        row[field] = str(relative / name)
                shutil.copy2(states_path.parent / 'result.json', folder / 'result.json')
                link(states_path.parent / 'joints.npz', folder / 'joints.npz')
                row['joints_path'] = str(relative / 'joints.npz')
                boundary = None
            else:
                result = None
                prefix = 'imprint-failure-pilot-20261105/fixed-labels' if dataset == PILOT else 'imprint-cover-fixed-20261105'
                bp = BASE / 'annotations' / prefix / dataset / 'boundaries' / (key + '.json')
                boundary = read(bp) if bp.exists() else None
                if boundary:
                    shutil.copy2(bp, folder / 'boundaries.json')
                    row['subtask_boundary_path'] = str(relative / 'boundaries.json')
            tag = intervention(scripted, result, boundary, states)
            row['intervention_path'] = None
            if tag:
                save(folder / 'intervention.json', tag)
                row['intervention_path'] = str(relative / 'intervention.json')
            row['label_status'] = ('pending_fixed_progress' if row['progress_path'] is None else
                                   'scripted_verified' if scripted else 'auto_boundary_needs_review')
            all_rows.append(row)
            if row['progress_path'] and not row['duplicate_of'] and not original.get('duplicate_failure'):
                labeled.append(row)
    for name, data in [('metadata.jsonl', all_rows), ('supervised.jsonl', labeled)]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    summary = dict(total=len(all_rows), labeled_deduplicated=len(labeled),
                   by_source=Counter(r['source_dataset'] for r in all_rows),
                   supervised_by_source=Counter(r['trajectory_source'] for r in labeled),
                   pending_progress=sum(r['progress_path'] is None for r in all_rows),
                   duplicates=sum(bool(r['duplicate_of']) or bool(r.get('duplicate_failure')) for r in all_rows))
    rules = VIEWERS / SCRIPTED / 'code' / 'rules'
    assert hashlib.sha256((rules / 'cover_blocks_weights.json').read_bytes()).hexdigest() == WEIGHTS_SHA
    shutil.copytree(rules, output / 'code' / 'rules')
    save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    target = Path(sys.argv[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with tempfile.TemporaryDirectory(prefix='.merge-s01-', dir=target.parent) as temp:
        staging = Path(temp) / 'dataset'
        staging.mkdir()
        merge(staging)
        staging.rename(target)
