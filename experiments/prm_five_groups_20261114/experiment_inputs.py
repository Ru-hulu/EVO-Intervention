"""Frozen inputs/readiness for all follow-up conditions; never invent labels."""
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import time

from .eval_common import EVAL, PLAN, sampling_indices
from .prepare_cache import BASE, DATA, RECOVER_PROMPTS, progress_values, save, sha
from .recover_metrics import validate_events
from .stream_manifests import digest, freeze
from .stream_runtime import load_inputs, validate_entry

MODELS = ('robometer-4b', 'robobrain25-4b', 'dopamine-3b', 'procvlm-2b',
          'r2vlm-alfred-rl', 'vlac-2b', 'forest')
MEMORY_MODELS = ('procvlm-2b', 'r2vlm-alfred-rl', 'forest', 'robometer-4b')
CONDITIONS = {
    2: ('no_memory',) + tuple(f'{v}-seed{s}' for v in ('static_only', 'rewind_only') for s in (42, 43, 44)),
    3: ('recover',), 4: ('own_prompt', 'swapped_prompt'),
    5: tuple(f'fps{f}' for f in (30, 20, 10, 5, 2))}


class InputBlocked(ValueError):
    def __init__(self, reasons):
        self.reasons = reasons
        super().__init__(json.dumps(reasons, ensure_ascii=False))


def annotation_path(group, key, evaluation=EVAL):
    relative = Path(key)
    assert not relative.is_absolute() and '..' not in relative.parts
    return Path(evaluation)/'annotations'/f'group{group}'/relative.with_name(relative.name+'.json')


def baseline(evaluation=EVAL, plan=PLAN):
    path = Path(evaluation)/'group1-manifest.json'
    value = json.loads(path.read_text()); assert value['plan_sha256'] == sha(plan)
    return value, sha(path)


def regular_entry(row, condition):
    entry = dict(key=row['key'], suite=row['suite'], task_id=row['task_id'], condition=condition,
                 input_fps=5, memory_enabled=True, progress_sha256=row['progress_sha256'],
                 source_frame_indices=row['frame_indices'], playback_timestamps=row['timestamps'],
                 target_pt=row['target_pt'], input_issues=row.get('input_issues', []),
                 all_views_share_source_frame_map=True)
    entry['map_sha256'] = digest(entry)
    return entry


def counterfactual_truth(label, row, prompt):
    assert label['schema'] == 'prm-instruction-pt-v1' and label['review_status'] == 'verified'
    assert label['key'] == row['key'] and label['task_prompt'] == prompt
    assert label['video_sha256'] == row['video_hashes'][0]
    assert label['frame_indices'] == list(range(row['frames']))
    values = label['target_pt']
    assert len(values) == row['frames'] and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in values)
    assert label['annotation_method'] and label['annotation_provenance']
    for evidence in label['annotation_provenance']:
        assert sha(evidence['path']) == evidence['sha256']
    return values


def template(group, row, prompt=None, evaluation=EVAL):
    path = annotation_path(group, row['key'], evaluation)
    target = Path(evaluation)/'annotation-templates'/path.relative_to(Path(evaluation)/'annotations')
    value = dict(schema='prm-instruction-pt-v1' if group == 4 else 'prm-recover-events-v1',
                 review_status='unreviewed', key=row['key'], video_sha256=row['video_hashes'][0],
                 total_frames=row['frames'], annotation_provenance=[])
    if group == 4:
        value.update(task_prompt=prompt, frame_indices=list(range(row['frames'])), target_pt=None,
                     annotation_method=None, note='Supply independently verified PT for this exact instruction; do not copy own-prompt PT.')
    else:
        value.update(progress_sha256=row['progress_sha256'], success=None, failure_onset_frame=None,
                     trr_role=None, abnormal_plateaus=[],
                     note='Use explicit outcome/failure/abnormal-interval annotations; PT=1 alone is not a success label.')
    if not target.exists():
        save(target, value)
    return path


def recover_rows(data=DATA, setup=BASE, plan=PLAN):
    fixed = json.loads(Path(plan).read_text())
    root = Path(data)/'Recover'; manifest_path = root/'manifest.json'
    assert sha(manifest_path) == fixed['datasets']['suites']['Recover']['manifest_sha256']
    manifest = json.loads(manifest_path.read_text()); rows, missing = [], []
    for task in manifest['tasks']:
        if task['task_id'] not in fixed['datasets']['recover_task_ids']:
            continue
        meta = root/task['trajectory_index']
        assert sha(meta) == task['trajectory_index_sha256']
        by_id = {r['sample_id']: r for r in map(json.loads, meta.read_text().splitlines())}
        for key in task['val']:
            r = by_id[key]; stem = f"episode_{int(r['episode_index']):06d}"
            progress = meta.parent/(r.get('progress_path') or f'annotation/{stem}.progress.json')
            if task['task_id'] == 'R05' and not progress.exists():
                progress = Path(setup)/'labels/Recover/R05'/f'{stem}.progress.csv'
            if not progress.exists():
                missing.append(dict(key=f"Recover/{task['task_id']}/{key}", reason='missing_progress', path=str(progress)))
                continue
            views = r['video_views']; order = ['main_view', 'wrist_view', None] if 'wrist_view' in views else ['main_view', 'left_view', 'right_view']
            videos = [str(meta.parent/views[v]['video_path']) if v in views else None for v in order]
            hashes = [views[v]['video_sha256'] if v in views else None for v in order]
            rows.append(dict(key=f"Recover/{task['task_id']}/{key}", suite='Recover', task_id=task['task_id'],
                sample_id=key, split='val', frames=r['total_frames'], fps=r['fps'], videos=videos,
                video_hashes=hashes, progress_path=str(progress), progress_sha256=sha(progress),
                task_prompt=RECOVER_PROMPTS[task['task_id']], metadata_path=str(meta), metadata_sha256=sha(meta)))
    assert len(rows)+len(missing) == 34
    if missing:
        raise InputBlocked(missing)
    pour_proof = json.loads((Path(setup)/'recover-pour-labels/verification.json').read_text())
    assert pour_proof.get('status') == 'passed', 'Supplemental PT verification is required'
    assert (pour_proof['episodes'],pour_proof['train'],pour_proof['val'])==(35,28,7)
    assert pour_proof['all_native_frames_verified'] and pour_proof['recover_manifest_unchanged']
    assert pour_proof['plan_sha256']==sha(plan)
    import av
    for row in rows:
        rates, issues = [], []
        for view, (path, expected) in enumerate(zip(row['videos'], row['video_hashes'])):
            if path:
                assert sha(path) == expected
                with av.open(path) as video:
                    stream = video.streams.video[0]; rates.append(stream.average_rate)
                    if stream.frames and stream.frames != row['frames']:
                        issues.append(dict(reason='view_frame_count_mismatch', view=view, path=path,
                                           expected=row['frames'], header_frames=stream.frames))
        assert len(set(rates)) == 1
        row['source_rate'] = str(rates[0]); row['input_issues'] = issues
        row['frame_indices'] = sampling_indices(row['frames'], rates[0])
        row['timestamps'] = [i/float(rates[0]) for i in row['frame_indices']]
        truth = progress_values(row); row['target_pt'] = [truth[i] for i in row['frame_indices']]
    return sorted(rows, key=lambda r: r['key'])


def prepare_condition(group, condition, evaluation=EVAL, plan=PLAN, data=DATA, setup=BASE):
    assert group in CONDITIONS and condition in CONDITIONS[group]
    evaluation = Path(evaluation)
    if group in (2, 5):
        manifest, rows, value_sha = load_inputs(group, condition, 'procvlm-2b', evaluation, plan)
        return manifest, rows, value_sha
    destination = evaluation/'conditions'/f'group{group}'/f'{condition}.json'
    if destination.exists():
        value = json.loads(destination.read_text())
        assert value['plan_sha256'] == sha(plan)
        assert value['baseline_manifest_sha256'] == sha(evaluation/'group1-manifest.json')
        for row in value['source_rows']:
            assert sha(row['progress_path']) == row['progress_sha256']
        return value, {r['key']: r for r in value['source_rows']}, sha(destination)
    original, original_sha = baseline(evaluation, plan)
    if group == 3:
        rows = recover_rows(data, setup, plan)
    else:
        rows = [dict(r) for r in original['episodes'] if r['task_id'] in ('O12', 'S13')]
        assert Counter(r['task_id'] for r in rows) == {'O12': 7, 'S13': 7}
        prompts = {task: {r['task_prompt'] for r in rows if r['task_id'] == task} for task in ('O12', 'S13')}
        assert all(len(v) == 1 for v in prompts.values())
        if condition == 'swapped_prompt':
            missing = []
            for row in rows:
                prompt = next(iter(prompts['S13' if row['task_id'] == 'O12' else 'O12']))
                path = template(4, row, prompt, evaluation)
                if not path.exists():
                    missing.append(dict(key=row['key'], reason='missing_instruction_matched_pt', path=str(path)))
                    continue
                values = counterfactual_truth(json.loads(path.read_text()), row, prompt)
                row.update(task_prompt=prompt, original_progress_sha256=row['progress_sha256'],
                           progress_path=str(path), progress_sha256=sha(path),
                           target_pt=[values[i] for i in row['frame_indices']])
            if missing:
                raise InputBlocked(missing)
    entries = [regular_entry(r, condition) for r in rows]
    value = dict(schema='prm-followup-inputs-v1', group=group, condition=condition,
                 plan_sha256=sha(plan), baseline_manifest_sha256=original_sha,
                 expected_episodes=len(rows), expected_samples=sum(len(e['target_pt']) for e in entries),
                 by_suite=dict(Counter(r['suite'] for r in rows)), source_rows=rows, episodes=entries)
    freeze(destination, value)
    return value, {r['key']: r for r in rows}, sha(destination)


def verified_condition(group, condition, evaluation=EVAL, plan=PLAN):
    if group in (2, 5):
        return load_inputs(group, condition, 'procvlm-2b', evaluation, plan)
    path = Path(evaluation)/'conditions'/f'group{group}'/f'{condition}.json'
    value = json.loads(path.read_text())
    assert value['group'] == group and value['condition'] == condition
    assert value['plan_sha256'] == sha(plan)
    assert value['baseline_manifest_sha256'] == sha(Path(evaluation)/'group1-manifest.json')
    rows = {r['key']: r for r in value['source_rows']}
    assert len(rows) == len(value['episodes']) == value['expected_episodes']
    assert sum(len(e['target_pt']) for e in value['episodes']) == value['expected_samples']
    for entry in value['episodes']:
        row=rows[entry['key']]
        validate_entry(row, entry)
        assert entry['source_frame_indices']==row['frame_indices']
        assert entry['playback_timestamps']==row['timestamps']
        assert entry['target_pt']==row['target_pt']
        if group==4 and condition=='swapped_prompt':
            truth=counterfactual_truth(json.loads(Path(row['progress_path']).read_text()),row,row['task_prompt'])
        else:
            truth=progress_values(row)
        assert entry['target_pt']==[truth[i] for i in entry['source_frame_indices']]
    return value, rows, sha(path)


def event_annotations(row, evaluation=EVAL):
    path = annotation_path(3, row['key'], evaluation)
    if not path.exists():
        template(3, row, evaluation=evaluation)
        return None, {'reason': 'missing_verified_event_annotation', 'path': str(path)}
    try:
        event = validate_events(json.loads(path.read_text()), row, progress_values(row))
    except (AssertionError,ValueError,KeyError,FileNotFoundError) as exc:
        # Outcome/event annotations are not model inputs. Invalid annotations
        # block these metrics, while valid PT still permits the VOC experiment.
        return None, {'path':str(path),'sha256':sha(path),'reason':'invalid_verified_event_annotation',
                      'error':f'{type(exc).__name__}: {exc}'}
    return event, {'path': str(path), 'sha256': sha(path)}


def prepare_all():
    status = {}
    for group, conditions in CONDITIONS.items():
        for condition in conditions:
            key = f'group{group}/{condition}'
            try:
                manifest, rows, value_sha = prepare_condition(group, condition)
                event_missing, event_proofs = [], {}
                if group == 3:
                    for row in rows.values():
                        event, proof = event_annotations(row)
                        event_proofs[row['key']] = proof
                        if event is None:
                            event_missing.append(dict(key=row['key'], **proof))
                status[key] = dict(status='ready', manifest_sha256=value_sha,
                                   expected_episodes=manifest['expected_episodes'],
                                   expected_samples=manifest['expected_samples'], event_metric_blockers=event_missing,
                                   event_annotation_revision=digest(event_proofs))
            except InputBlocked as exc:
                status[key] = dict(status='blocked_by_inputs', blockers=exc.reasons)
            except Exception as exc:
                status[key] = dict(status='blocked_by_validation', error=f'{type(exc).__name__}: {exc}')
    save(EVAL/'conditions/readiness.json', dict(updated=time.time(), conditions=status,
                                             implementation_complete=True, gpu_validation_passed=False))
    return status


if __name__ == '__main__':
    print(json.dumps(prepare_all(), ensure_ascii=False))
