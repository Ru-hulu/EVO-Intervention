"""Resumable official-model worker for frozen group-2/5 streams.

Not registered with the dispatcher yet. GPU entry requires an exact live
dispatcher admission; CPU tests exercise persistence without loading a model.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

from .eval_common import EVAL, OUT, PLAN
from .official_baseline import (
    KINDS, OLD, engine, input_blockers, model_audit, native_module)
from .prepare_cache import save, sha
from .stream_images import prepare
from .stream_manifests import digest
from .stream_runtime import (
    StreamSession, install_proc_memory_policy, load_inputs, mapped_episode)


def job_name(model, group, condition, shard):
    assert model in KINDS and group in (2, 5) and 0 <= shard < 4
    allowed = ({'no_memory'} | {f'{v}-seed{s}' for v in ('static_only', 'rewind_only')
                              for s in (42, 43, 44)}) if group == 2 else {
                                  f'fps{fps}' for fps in (30, 20, 10, 5, 2)}
    assert condition in allowed
    assert group != 2 or model in ('procvlm-2b', 'r2vlm-alfred-rl')
    return f'{model}-group{group}-{condition}-{shard}'


def check_admission(receipt, dispatcher, name, model, group, condition, shard,
                    gpu, pid, memory_fraction, now):
    """Do not allow this future worker to bypass the existing resource owner."""
    assert dispatcher['status'] == 'running' and now - dispatcher['updated'] < 30
    active = dispatcher['active'][str(gpu)]
    for record in (receipt, active):
        assert record['name'] == name and record['pid'] == pid and record['gpu'] == gpu
        assert record['model'] == model and record['group'] == group and record['shard'] == shard
        assert record['condition'] == condition and record['batch'] == 1
        assert record['memory_fraction'] == memory_fraction
        assert record['module'] == 'experiments.prm_five_groups_20261114.official_stream'
    assert receipt['dispatcher_pid'] == dispatcher['pid']
    assert receipt['first_group_priority_cleared'] is True
    assert receipt['resource_admission_passed'] is True
    assert 0 <= now - receipt['started'] < 30


def paths(output, group, condition, model, key):
    relative = Path(key)
    assert not relative.is_absolute() and '..' not in relative.parts
    root = Path(output)/f'group{group}'/condition/model
    # Retain the source hierarchy; do not flatten potentially ambiguous keys.
    relative = relative.with_name(relative.name+'.json')
    return root/'episodes'/relative, root/'partials'/relative


def pause_requested(name, gpu, pid, evaluation=EVAL, now=None):
    """Save a prefix if the resource owner disappears or withdraws this slot."""
    evaluation = Path(evaluation)
    if (evaluation/f'pause-{name}').exists():
        return True
    try:
        dispatcher = json.loads((evaluation/'dispatcher-status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return True
    active = dispatcher.get('active', {}).get(str(gpu), {})
    return (dispatcher.get('status') != 'running'
            or (time.time() if now is None else now) - dispatcher.get('updated', 0) > 90
            or active.get('name') != name or active.get('pid') != pid)


def run_episode(native, predictor, row, entry, identity, image_root, result_path,
                partial_path, audit_path, should_pause, on_sample=lambda _: None):
    """Persist each answer before advancing, and never repeat a saved prefix."""
    kind = KINDS[identity['model']]
    episode = mapped_episode(row, entry, image_root, kind)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        session = StreamSession(kind, episode, entry, identity, native.INITIAL_HISTORY, result)
        assert session.complete and result['stream_map'] == entry
        assert result['metrics'] == session.result()['metrics']
        assert result['source_video_sha256'] == row['video_hashes']
        assert result['progress_sha256'] == row['progress_sha256']
        audit = Path(result['initialization_audit'])
        assert sha(audit) == result['initialization_audit_sha256']
        assert json.loads(audit.read_text())['identity'] == identity
        for proof in result['initialization_audits']:
            assert sha(proof['path']) == proof['sha256']
            assert json.loads(Path(proof['path']).read_text())['identity'] == identity
        return 'existing', result
    prefix = json.loads(partial_path.read_text()) if partial_path.exists() else None
    session = StreamSession(kind, episode, entry, identity, native.INITIAL_HISTORY, prefix)
    if prefix is not None:
        for proof in prefix['initialization_audits']:
            audit = Path(proof['path'])
            assert sha(audit) == proof['sha256']
            assert json.loads(audit.read_text())['identity'] == identity
        audits = list(prefix['initialization_audits'])
    else:
        audits = []
    current_audit = {'path': str(audit_path), 'sha256': sha(audit_path)}
    if current_audit not in audits:
        audits.append(current_audit)
    while not session.complete:
        if should_pause():
            # Includes VLAC's native first value, even before its first call.
            save(partial_path, dict(session.checkpoint(), initialization_audits=audits))
            return 'paused', session.checkpoint()
        values, answers = predictor.infer([session.request()])
        assert len(values) == len(answers) == 1
        session.accept(values[0], answers[0])
        save(partial_path, dict(session.checkpoint(), initialization_audits=audits))
        on_sample(session.checkpoint())
    result = dict(session.result(), schema='prm-official-stream-v1',
                  suite=row['suite'], task_id=row['task_id'], input_fps=entry['input_fps'],
                  source_video_sha256=row['video_hashes'], progress_sha256=row['progress_sha256'],
                  uses_future_endpoint_reference=kind == 'grm',
                  initialization_audit=str(audit_path),
                  initialization_audit_sha256=current_audit['sha256'],
                  initialization_audits=audits, finished=time.time())
    # The job lock ensures a single writer for this fixed four-way partition.
    assert not result_path.exists(), 'Refuse to overwrite a completed trajectory'
    save(result_path, result)
    partial_path.unlink(missing_ok=True)
    return 'completed', result


def evaluate(args):
    name = job_name(args.model, args.group, args.condition, args.shard)
    gpu_text = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert gpu_text in ('0', '1', '2', '3'), 'Exactly one dispatcher-assigned GPU required'
    gpu, pid = int(gpu_text), os.getpid()
    admission_path = EVAL/'dispatch-launches'/f'{name}-{pid}.json'
    # Popen returns before the parent atomically publishes its launch record.
    deadline = time.monotonic() + 10
    while True:
        if admission_path.exists():
            receipt = json.loads(admission_path.read_text())
            dispatcher = json.loads((EVAL/'dispatcher-status.json').read_text())
            if dispatcher.get('active', {}).get(str(gpu), {}).get('pid') == pid:
                break
        if time.monotonic() >= deadline:
            raise RuntimeError('No live dispatcher admission; no GPU model was loaded')
        time.sleep(.2)
    check_admission(receipt, dispatcher, name, args.model, args.group, args.condition,
                    args.shard, gpu, pid, args.memory_fraction, time.time())
    assert Path(f'/proc/{dispatcher["pid"]}').exists()
    jobs = EVAL/'jobs'; jobs.mkdir(exist_ok=True)
    lock = (jobs/f'{name}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(model=args.model, group=args.group, condition=args.condition,
                 shard=args.shard, shards=4, gpu=gpu, pid=pid, started=time.time(),
                 completed_episodes=0, completed_samples=0, batch=1,
                 memory_fraction=args.memory_fraction)
    def update(**changes):
        state.update(changes, updated=time.time()); save(jobs/f'{name}.json', state)
    try:
        update(status='verifying_inputs_and_weights')
        manifest, sources, manifest_sha = load_inputs(args.group, args.condition, args.model)
        selected = manifest['episodes'][args.shard::4]
        kind = KINDS[args.model]
        blocked = [{'key': e['key'], 'issues': input_blockers(sources[e['key']], kind)}
                   for e in selected if input_blockers(sources[e['key']], kind)]
        update(total_episodes=len(selected), total_samples=sum(len(e['target_pt']) for e in selected),
               input_blockers=blocked, expected_episodes=manifest['expected_episodes'])
        ref, hashes = model_audit(args.model)
        native = native_module(); cfg = dict(native.MODELS[args.model], batch=1)
        assert cfg['kind'] == kind and cfg['path'] == ref['path']
        if kind == 'proc':
            install_proc_memory_policy(native)
        identity = dict(model=args.model, group=args.group, condition=args.condition,
                        official_revision=ref['revision'], checkpoint=ref['path'],
                        official_files_sha256=digest(hashes), native_config=cfg,
                        native_worker_sha256=sha(OLD/'worker.py'),
                        native_common_sha256=sha(OLD/'common.py'),
                        prompt_sha256={p: sha(OLD/p) for p in
                                       ('grm_prompt.txt', 'proc_prompt.txt', 'r2_prompt.txt')},
                        plan_sha256=sha(PLAN), manifest_sha256=manifest_sha,
                        baseline_manifest_sha256=manifest['baseline_manifest_sha256'],
                        batch=1, memory_fraction=args.memory_fraction)
        if pause_requested(name, gpu, pid):
            update(status='paused_by_resource_guard'); return
        import torch
        torch.set_num_threads(2); torch.manual_seed(42)
        if kind == 'vlac':
            torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        update(status='loading_model')
        predictor = engine(native, cfg, args.memory_fraction)
        audit_path = jobs/'stream-initializations'/f'{name}-{pid}-{time.time_ns()}.json'
        save(audit_path, dict(identity=identity, official_file_sha256=hashes,
                             adapter_loaded=False, native_model_loaded=True,
                             worker_sha256=sha(__file__), torch=torch.__version__,
                             created=time.time(), gpu=gpu))
        update(status='running', initialization_audit=str(audit_path))
        new_samples, started = 0, time.time()
        def progress(prefix):
            nonlocal new_samples
            new_samples += 1
            update(phase='inference', samples_in_episode=len(prefix['predictions']),
                   new_samples=new_samples, samples_per_second=new_samples/max(time.time()-started, 1))
        blocked_keys = {r['key'] for r in blocked}
        for entry in selected:
            if entry['key'] in blocked_keys:
                continue
            if pause_requested(name, gpu, pid):
                update(status='paused_by_resource_guard'); return
            row = sources[entry['key']]
            assert sha(row['progress_path']) == row['progress_sha256']
            update(episode=entry['key'], phase='decode_and_verify_images')
            for view, path in enumerate(row['videos']):
                if path and (view == 0 or kind == 'grm'):
                    assert sha(path) == row['video_hashes'][view], path
            root = prepare(row, entry, main_only=kind != 'grm')
            native.image.cache_clear()
            result, partial = paths(OUT, args.group, args.condition, args.model, row['key'])
            status, saved = run_episode(native, predictor, row, entry, identity, root,
                                        result, partial, audit_path,
                                        lambda: pause_requested(name, gpu, pid), progress)
            if status == 'paused':
                update(status='paused_by_resource_guard'); return
            update(completed_episodes=state['completed_episodes']+1,
                   completed_samples=state['completed_samples']+len(saved['predictions']))
        update(status='completed_with_input_blockers' if blocked else 'completed', finished=time.time(),
               aggregation_pending=True)
    except BaseException:
        update(status='failed', error=traceback.format_exc()); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=list(KINDS))
    parser.add_argument('--group', type=int, choices=[2, 5], required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--memory-fraction', type=float, default=.30)
    args = parser.parse_args()
    assert 0 <= args.shard < 4 and 0 < args.memory_fraction <= .35
    evaluate(args)
