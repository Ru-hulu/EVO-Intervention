"""Seven-model worker for groups 2--5, with exact durable prefix recovery."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import traceback

from .eval_common import EVAL, OUT, PLAN, trajectory_metrics
from .experiment_inputs import MODELS, MEMORY_MODELS, CONDITIONS, verified_condition
from .official_stream import pause_requested, paths
from .prepare_cache import save, sha
from .stream_images import prepare
from .stream_manifests import digest, score_fps_stream
from .stream_runtime import validate_entry


def name_for(model, group, condition, shard, suite=None):
    assert model in MODELS and group in CONDITIONS and condition in CONDITIONS[group]
    assert 0 <= shard < 4 and (group != 2 or model in MEMORY_MODELS)
    assert (model == 'forest') == (suite is not None)
    if suite is not None:
        assert suite in (('Recover',) if group == 3 else ('State', 'Order') if group == 4 else ('State', 'Order', 'Count'))
    return f'{model}' + (f'-{suite.lower()}' if suite else '') + f'-group{group}-{condition}-{shard}'


def required_issues(row, model):
    return [issue for issue in row.get('input_issues', [])
            if model in ('robometer-4b', 'robobrain25-4b', 'dopamine-3b', 'forest') or issue.get('view') == 0]


def check_prefix(record, identity, entry, complete=False):
    assert record['identity'] == identity, 'Prediction provenance differs'
    n = len(record['predictions'])
    assert n <= len(entry['target_pt']) and (not complete or n == len(entry['target_pt']))
    for field in ('source_frame_indices', 'playback_timestamps', 'target_pt'):
        assert record[field] == entry[field][:n], field
    assert len(record['raw_responses']) == len(record['history_lengths']) == n
    assert all(v is None or (type(v) in (float, int) and math.isfinite(v)) for v in record['predictions'])
    model = identity['model']
    for i, raw in enumerate(record['raw_responses']):
        if model in ('robometer-4b', 'forest') or (model == 'vlac-2b' and i == 0):
            assert raw is None or isinstance(raw, str)
        else:
            assert isinstance(raw, str)
    if model == 'vlac-2b' and n:
        assert record['predictions'][0] == 0. and record['raw_responses'][0] is None
    for proof in record['initialization_audits']:
        assert sha(proof['path']) == proof['sha256']
        audit = json.loads(Path(proof['path']).read_text())
        assert audit['model_signature'] == identity['model_signature']
        assert audit['plan_sha256'] == identity['plan_sha256']
    if complete:
        assert record['stream_map'] == entry
    return n


def metrics(predictions, entry):
    return (score_fps_stream(predictions, entry) if 'common_scoring_indices' in entry else
            {'full_stream': trajectory_metrics(predictions, entry['target_pt'], entry['playback_timestamps'])})


def execute_episode(backend, row, entry, identity, root, result_path, partial_path,
                    audit_path, should_pause, on_sample=lambda _: None):
    validate_entry(row, entry)
    if result_path.exists():
        record = json.loads(result_path.read_text())
        check_prefix(record, identity, entry, complete=True)
        assert record['metrics'] == metrics(record['predictions'], entry)
        return 'existing', record
    if partial_path.exists():
        record = json.loads(partial_path.read_text()); check_prefix(record, identity, entry)
    else:
        record = dict(identity=identity, predictions=[], raw_responses=[], history_lengths=[],
                      source_frame_indices=[], playback_timestamps=[], target_pt=[], initialization_audits=[])
    proof = dict(path=str(audit_path), sha256=sha(audit_path))
    if proof not in record['initialization_audits']:
        record['initialization_audits'].append(proof)
    if not backend.begin(row, entry, root, record, should_pause):
        save(partial_path, record); return 'paused', record
    while len(record['predictions']) < len(entry['target_pt']):
        if should_pause():
            save(partial_path, record); return 'paused', record
        i = len(record['predictions'])
        value, raw, history_length = backend.predict(i, record['predictions'], record['raw_responses'])
        value = float(value) if value is not None and math.isfinite(value) else None
        record['predictions'].append(value); record['raw_responses'].append(raw)
        record['history_lengths'].append(history_length)
        for field in ('source_frame_indices', 'playback_timestamps', 'target_pt'):
            record[field].append(entry[field][i])
        save(partial_path, record)
        on_sample(record)
    record.update(schema='prm-seven-model-followup-v1', stream_map=entry,
                  metrics=metrics(record['predictions'], entry), finished=time.time())
    assert not result_path.exists(), 'Never overwrite completed predictions'
    save(result_path, record); partial_path.unlink(missing_ok=True)
    return 'completed', record


def admitted(args, name):
    gpu_text = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert gpu_text in ('0', '1', '2', '3')
    gpu, pid = int(gpu_text), os.getpid()
    path = EVAL/'dispatch-launches'/f'{name}-{pid}.json'
    deadline = time.monotonic()+25
    while time.monotonic() < deadline:
        if path.exists():
            receipt = json.loads(path.read_text())
            dispatcher = json.loads((EVAL/'dispatcher-status.json').read_text())
            active = dispatcher.get('active', {}).get(str(gpu), {})
            if active.get('pid') == pid:
                for r in (receipt, active):
                    assert r['name'] == name and r['gpu'] == gpu and r['pid'] == pid
                    assert r['model'] == args.model and r['group'] == args.group
                    assert r['condition'] == args.condition and r['suite'] == args.suite and r['shard'] == args.shard
                    assert r['module'] == 'experiments.prm_five_groups_20261114.experiment_worker'
                    assert r['batch'] == 1 and r['memory_fraction'] == args.memory_fraction
                assert receipt['dispatcher_pid'] == dispatcher['pid']
                assert receipt['resource_admission_passed'] and receipt['first_group_priority_cleared']
                assert dispatcher['status'] == 'running' and time.time()-dispatcher['updated'] < 60
                assert 0 <= time.time()-receipt['started'] < 60
                assert Path(f'/proc/{dispatcher["pid"]}').exists()
                return gpu
        time.sleep(.2)
    raise RuntimeError('No matching live dispatcher admission; no model loaded')


def evaluate(args):
    from .experiment_backends import load_backend
    name = name_for(args.model, args.group, args.condition, args.shard, args.suite)
    gpu = admitted(args, name)
    jobs = EVAL/'jobs'; jobs.mkdir(exist_ok=True)
    lock = (jobs/f'{name}.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(name=name, model=args.model, group=args.group, condition=args.condition, suite=args.suite,
                 shard=args.shard, pid=os.getpid(), gpu=gpu, started=time.time(), batch=1,
                 memory_fraction=args.memory_fraction, completed_episodes=0, completed_samples=0)
    def update(**changes):
        state.update(changes, updated=time.time()); save(jobs/f'{name}.json', state)
    paused = lambda: pause_requested(name, gpu, os.getpid())
    try:
        update(status='verifying_inputs')
        manifest, rows, manifest_sha = verified_condition(args.group, args.condition)
        candidates = [e for e in manifest['episodes'] if args.suite is None or e['suite'] == args.suite]
        selected = candidates[args.shard::4]
        blockers = [dict(key=e['key'], issues=required_issues(rows[e['key']], args.model))
                    for e in selected if required_issues(rows[e['key']], args.model)]
        update(total_episodes=len(selected), total_samples=sum(len(e['target_pt']) for e in selected), input_blockers=blockers)
        if paused():
            update(status='paused_by_resource_guard'); return
        update(status='loading_model')
        backend = load_backend(args.model, args.suite, args.condition != 'no_memory', args.memory_fraction)
        model_signature = digest(backend.audit)
        audit = dict(backend.audit, model_signature=model_signature, plan_sha256=sha(PLAN),
                     manifest_sha256=manifest_sha, group=args.group, condition=args.condition,
                     memory_enabled=args.condition != 'no_memory', worker_code_sha256=sha(__file__))
        audit_path = jobs/'followup-initializations'/f'{name}-{os.getpid()}-{time.time_ns()}.json'
        save(audit_path, audit); update(status='running', initialization_audit=str(audit_path))
        start, new_samples = time.time(), 0
        def progress(record):
            nonlocal new_samples
            new_samples += 1
            update(phase='inference', samples_in_episode=len(record['predictions']),
                   new_samples=new_samples, samples_per_second=new_samples/max(time.time()-start, 1))
        for entry in selected:
            row = rows[entry['key']]
            if required_issues(row, args.model):
                continue
            if paused():
                update(status='paused_by_resource_guard'); return
            update(phase='prepare_images', episode=row['key'])
            assert sha(row['progress_path']) == row['progress_sha256']
            main_only = args.model in ('procvlm-2b', 'r2vlm-alfred-rl', 'vlac-2b')
            for view, path in enumerate(row['videos']):
                if path and (view == 0 or not main_only):
                    assert sha(path) == row['video_hashes'][view]
            root = prepare(row, entry, main_only=main_only)
            identity = dict(model=args.model, group=args.group, condition=args.condition,
                suite=row['suite'], task_id=row['task_id'], key=row['key'], input_fps=entry['input_fps'],
                plan_sha256=sha(PLAN), manifest_sha256=manifest_sha, map_sha256=entry['map_sha256'],
                model_signature=model_signature, task_prompt=row['task_prompt'],
                progress_sha256=row['progress_sha256'], source_video_sha256=row['video_hashes'],
                memory_enabled=entry['memory_enabled'], uses_future_endpoint_reference=args.model in ('robobrain25-4b', 'dopamine-3b'),
                batch=1, memory_fraction=args.memory_fraction)
            result_path, partial_path = paths(OUT, args.group, args.condition, args.model, row['key'])
            status, record = execute_episode(backend, row, entry, identity, root, result_path,
                partial_path, audit_path, paused, progress)
            if status == 'paused':
                update(status='paused_by_resource_guard'); return
            canary = EVAL/'canaries'/f'{name}.json'
            if not canary.exists():
                generated = record['predictions'][1:] if args.model == 'vlac-2b' else record['predictions']
                assert any(v is not None for v in generated), 'Canary has no finite generated predictions; inspect parser/model'
                save(canary, dict(status='passed', model=args.model, group=args.group, condition=args.condition,
                    suite=args.suite, key=row['key'], complete_real_episode=True, model_signature=model_signature,
                    manifest_sha256=manifest_sha, result_path=str(result_path), result_sha256=sha(result_path), at=time.time()))
            update(completed_episodes=state['completed_episodes']+1,
                   completed_samples=state['completed_samples']+len(record['predictions']))
        update(status='completed_with_input_blockers' if blockers else 'completed', finished=time.time())
    except BaseException:
        update(status='failed', error=traceback.format_exc()); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=MODELS); p.add_argument('--group', type=int, choices=[2, 3, 4, 5], required=True)
    p.add_argument('--condition', required=True); p.add_argument('--suite'); p.add_argument('--shard', type=int, required=True)
    p.add_argument('--memory-fraction', type=float, default=.30)
    a = p.parse_args(); assert 0 < a.memory_fraction <= .35
    evaluate(a)
