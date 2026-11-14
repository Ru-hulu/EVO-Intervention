import json

import pytest

from experiments.prm_five_groups_20261114.official_stream import (
    check_admission, job_name, paths, pause_requested, run_episode)
from experiments.prm_five_groups_20261114.prepare_cache import save
from experiments.prm_five_groups_20261114.stream_manifests import playback_map, fps_map
from experiments.prm_five_groups_20261114.test_stream_runtime import identity, source


class NativeStub:
    INITIAL_HISTORY = 'initial'


class PredictorStub:
    def __init__(self, fail_at=None):
        self.calls = []; self.fail_at = fail_at

    def infer(self, requests):
        assert len(requests) == 1
        episode, index, history = requests[0]
        assert set(episode) == {'task', 'frames', '_memory_enabled'}
        assert all(set(frame) == {'image'} for frame in episode['frames'])
        if index == self.fail_at:
            raise RuntimeError('simulated worker failure')
        self.calls.append((index, history))
        return [None if index == 21 else .01*index], [f'<think>step {index}</think>']


def context(tmp_path, model='r2vlm-alfred-rl', variant='rewind_only'):
    row = source(); row['video_hashes'] = ['main-sha', 'left-sha', 'right-sha']
    entry = (fps_map(row, {i: i/180 for i in range(181)}, 20) if variant == 'fps20'
             else playback_map(row, variant, None if variant == 'no_memory' else 42))
    ident = identity(model); audit = tmp_path/'audit.json'
    save(audit, {'identity': ident, 'native_model_loaded': True, 'adapter_loaded': False})
    result, partial = paths(tmp_path, 5 if variant == 'fps20' else 2, variant, model, row['key'])
    return row, entry, ident, result, partial, audit


@pytest.mark.parametrize('variant', ['no_memory', 'static_only', 'rewind_only'])
def test_pause_resume_persists_each_answer_and_never_repeats_prefix(tmp_path, variant):
    row, entry, ident, result, partial, audit = context(tmp_path, variant=variant)
    model = PredictorStub()
    def execute(model, pause):
        return run_episode(NativeStub, model, row, entry, ident, '/images', result, partial, audit, pause)
    status, prefix = execute(model, lambda: len(model.calls) == 23)
    assert status == 'paused' and not result.exists() and partial.exists()
    assert [i for i, _ in model.calls] == list(range(23))
    saved_prefix = json.loads(partial.read_text())
    resumed = PredictorStub()
    status, saved = execute(resumed, lambda: False)
    assert status == 'completed' and not partial.exists()
    assert resumed.calls[0] == (23, 'initial' if variant == 'no_memory' else 'step 22')
    assert [i for i, _ in resumed.calls] == list(range(23, len(entry['target_pt'])))
    assert saved['predictions'][:23] == saved_prefix['predictions']
    assert saved['raw_responses'][:23] == saved_prefix['raw_responses']
    assert saved['predictions'][21] is None and saved['metrics']['full_stream']['pearson'] is None
    original_bytes = result.read_bytes()
    unused = PredictorStub()
    status, _ = execute(unused, lambda: False)
    assert status == 'existing' and unused.calls == [] and result.read_bytes() == original_bytes


def test_unknown_failure_keeps_last_durable_prefix_and_does_not_make_final(tmp_path):
    row, entry, ident, result, partial, audit = context(tmp_path)
    with pytest.raises(RuntimeError, match='simulated worker failure'):
        run_episode(NativeStub, PredictorStub(fail_at=7), row, entry, ident,
                    '/images', result, partial, audit, lambda: False)
    assert not result.exists()
    assert len(json.loads(partial.read_text())['predictions']) == 7


@pytest.mark.parametrize('field', ['source_frame_indices', 'target_pt', 'identity', 'audit'])
def test_corrupt_prefix_or_provenance_rejected_before_another_prediction(tmp_path, field):
    row, entry, ident, result, partial, audit = context(tmp_path)
    model = PredictorStub()
    run_episode(NativeStub, model, row, entry, ident, '/images', result, partial, audit,
                lambda: len(model.calls) == 2)
    if field == 'audit':
        save(audit, {'identity': ident, 'tampered': True})
    else:
        data = json.loads(partial.read_text())
        if field == 'identity':
            data[field]['official_revision'] = 'other'
        else:
            data[field][0] += 1
        save(partial, data)
    fresh = PredictorStub()
    with pytest.raises(AssertionError):
        run_episode(NativeStub, fresh, row, entry, ident, '/images', result, partial, audit, lambda: False)
    assert not fresh.calls and not result.exists()


def test_vlac_first_value_pause_and_common_grid_are_preserved(tmp_path):
    row, entry, ident, result, partial, audit = context(tmp_path, 'vlac-2b', 'fps20')
    model = PredictorStub()
    status, _ = run_episode(NativeStub, model, row, entry, ident, '/images', result,
                            partial, audit, lambda: True)
    assert status == 'paused' and model.calls == []
    assert json.loads(partial.read_text())['predictions'] == [0.]
    status, saved = run_episode(NativeStub, model, row, entry, ident, '/images', result,
                               partial, audit, lambda: False)
    assert model.calls[0][0] == 1 and status == 'completed'
    assert saved['predictions'][:3] == pytest.approx([0., .01, .0298])
    assert all(v is None for v in saved['predictions'][21:])
    assert saved['metrics']['common_integer_seconds']['interpolated'] is False


def test_admission_requires_exact_live_scheduler_and_priority_clearance():
    name = job_name('procvlm-2b', 2, 'no_memory', 0)
    record = dict(name=name, pid=12, gpu=2, model='procvlm-2b', group=2, condition='no_memory',
                  shard=0, batch=1, memory_fraction=.3,
                  module='experiments.prm_five_groups_20261114.official_stream',
                  dispatcher_pid=1, first_group_priority_cleared=True,
                  resource_admission_passed=True, started=100.)
    dispatcher = dict(status='running', pid=1, updated=100., active={'2': record})
    def check(r, d, now=101.):
        check_admission(r, d, name, 'procvlm-2b', 2, 'no_memory', 0, 2, 12, .3, now)
    check(record, dispatcher)
    for field, value in [('pid', 99), ('first_group_priority_cleared', False),
                         ('resource_admission_passed', False), ('group', 5), ('memory_fraction', .8)]:
        changed = dict(record, **{field: value})
        with pytest.raises(AssertionError):
            check(changed, dispatcher)
    with pytest.raises(AssertionError):
        check(record, dispatcher, now=140.)
    with pytest.raises(AssertionError):
        job_name('robobrain25-4b', 2, 'no_memory', 0)


def test_missing_stale_or_reassigned_owner_pauses_without_advancing(tmp_path):
    assert pause_requested('job', 2, 12, tmp_path, 101.)
    state = dict(status='running', updated=100., active={'2': dict(name='job', pid=12)})
    save(tmp_path/'dispatcher-status.json', state)
    assert not pause_requested('job', 2, 12, tmp_path, 101.)
    assert pause_requested('job', 2, 12, tmp_path, 201.)
    assert pause_requested('job', 2, 99, tmp_path, 101.)
    (tmp_path/'pause-job').touch()
    assert pause_requested('job', 2, 12, tmp_path, 101.)


def test_completed_result_tampering_is_rejected_and_not_overwritten(tmp_path):
    row, entry, ident, result, partial, audit = context(tmp_path)
    run_episode(NativeStub, PredictorStub(), row, entry, ident, '/images', result,
                partial, audit, lambda: False)
    data = json.loads(result.read_text()); data['target_pt'][0] += .1
    save(result, data); original = result.read_bytes(); model = PredictorStub()
    with pytest.raises(AssertionError):
        run_episode(NativeStub, model, row, entry, ident, '/images', result,
                    partial, audit, lambda: False)
    assert result.read_bytes() == original and model.calls == []
