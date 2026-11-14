from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.prm_five_groups_20261114.stream_manifests import (
    digest, fps_map, playback_map)
from experiments.prm_five_groups_20261114.stream_runtime import (
    StreamSession, install_proc_memory_policy, load_inputs, mapped_episode)
from experiments.prm_five_groups_20261114.prepare_cache import sha


def source():
    return {'key': 'State/S01/example', 'suite': 'State', 'task_id': 'S01',
            'frames': 181, 'source_rate': '30', 'frame_indices': list(range(0, 181, 6)),
            'timestamps': [i/30 for i in range(0, 181, 6)],
            'target_pt': [i/180 for i in range(0, 181, 6)],
            'progress_sha256': 'source-label-hash', 'input_issues': [],
            'task_prompt': 'Put the block away.', 'videos': ['main', 'left', 'right']}


def identity(model):
    return dict(model=model, official_revision='official-revision',
                plan_sha256='plan', manifest_sha256='frozen-stream')


def session(kind='r2', variant='rewind_only', prefix=None):
    row = source()
    entry = playback_map(row, variant, None if variant == 'no_memory' else 42)
    ep = mapped_episode(row, entry, Path('/images'), kind)
    model = {'r2': 'r2vlm-alfred-rl', 'proc': 'procvlm-2b'}[kind]
    return StreamSession(kind, ep, entry, identity(model), 'initial', prefix)


def test_native_requests_follow_playback_without_exposing_truth():
    row = source()
    entry = playback_map(row, 'rewind_only', 42)
    episode = mapped_episode(row, entry, '/images', 'r2')
    assert set(episode) == {'task', 'frames', '_memory_enabled'}
    assert all(set(f) == {'image'} for f in episode['frames'])
    assert [int(Path(f['image'][0]).stem.split('-')[0]) for f in episode['frames']] == entry['source_frame_indices']
    assert all(f['image'][1:] == [None, None] for f in episode['frames'])
    state = session()
    for i in range(len(entry['source_frame_indices'])):
        ep, index, history = state.request()
        assert index == i and history == ('initial' if i == 0 else f'step {i-1}')
        assert 'target_pt' not in ep and 'playback_timestamps' not in ep
        state.accept(i/100, f'<think>step {i}</think><answer>5</answer>')
    assert state.complete
    assert len(state.result()['predictions']) == len(entry['source_frame_indices'])


def test_proc_no_memory_supplies_one_image_and_preserves_native_other_modes():
    def original(kind, ep, i):
        if kind == 'proc':
            return [ep['frames'][max(0, j)]['image'][0] for j in range(i-7, i+1)]
        return ['native-reference-layout']
    native = SimpleNamespace(images_for=original)
    install_proc_memory_policy(native)
    off = session('proc', 'no_memory')
    on = session('proc', 'static_only')
    assert native.images_for('proc', off.episode, 12) == ['/images/00000072-0.png']
    assert len(native.images_for('proc', on.episode, 2)) == 8
    assert native.images_for('proc', on.episode, 2) == original('proc', on.episode, 2)
    assert native.images_for('grm', on.episode, 12) == ['native-reference-layout']


@pytest.mark.parametrize('variant', ['no_memory', 'static_only', 'rewind_only'])
def test_r2_resume_exactly_preserves_next_request_and_existing_predictions(variant):
    full = session(variant=variant)
    for i in range(22):
        full.accept(None if i == 21 else i/100,
                    'malformed but retained' if i == 21 else f'<think>memory {i}</think>')
    prefix = full.checkpoint()
    resumed = session(variant=variant, prefix=prefix)
    assert resumed.request() == full.request()
    assert resumed.request()[2] == ('initial' if variant == 'no_memory' else 'malformed but retained')
    while not full.complete:
        i = len(full.predictions)
        full.accept(i/100, f'<think>future {i}</think>')
        resumed.accept(i/100, f'<think>future {i}</think>')
    assert resumed.result() == full.result()
    assert prefix['predictions'] == full.predictions[:22]
    assert full.result()['metrics']['full_stream']['pearson'] is None
    # New episodes always reset, irrespective of the preceding episode's result.
    assert session(variant=variant).request()[2] == 'initial'


@pytest.mark.parametrize('field', ['identity', 'source_frame_indices', 'target_pt', 'raw_responses'])
def test_resume_rejects_wrong_checkpoint_or_corrupt_prefix(field):
    state = session()
    state.accept(.1, '<think>first</think>')
    prefix = state.checkpoint()
    if field == 'identity':
        prefix[field]['official_revision'] = 'different-weights'
    elif field == 'source_frame_indices':
        prefix[field][0] += 1
    elif field == 'target_pt':
        prefix[field][0] += .01
    else:
        prefix[field][0] = None
    with pytest.raises(AssertionError):
        session(prefix=prefix)


def test_vlac_native_recurrence_and_null_survive_resume_and_common_grid():
    row = source()
    entry = fps_map(row, {i: i/180 for i in range(181)}, 20)
    ep = mapped_episode(row, entry, '/images', 'vlac')
    state = StreamSession('vlac', ep, entry, identity('vlac-2b'), 'initial')
    assert state.predictions == [0.] and state.raw == [None]
    state.accept(.5, '50')
    state.accept(-.2, '-20')
    assert state.predictions == [0., .5, .4]
    state.accept(None, 'unknown')
    resumed = StreamSession('vlac', ep, entry, identity('vlac-2b'), 'initial', state.checkpoint())
    while not resumed.complete:
        resumed.accept(.5, '50')
    assert all(v is None for v in resumed.predictions[3:])
    result = resumed.result()
    assert result['metrics']['full_stream']['pearson'] is None
    assert result['metrics']['common_integer_seconds']['pearson'] is None
    assert result['metrics']['common_integer_seconds']['interpolated'] is False


def test_fps_scoring_never_hides_full_stream_invalid_predictions():
    row = source()
    entry = fps_map(row, {i: i/180 for i in range(181)}, 20)
    ep = mapped_episode(row, entry, '/images', 'proc')
    state = StreamSession('proc', ep, entry, identity('procvlm-2b'), 'initial')
    with pytest.raises(RuntimeError):
        state.result()
    for i, value in enumerate(entry['target_pt']):
        state.accept(float('nan') if i == 1 else value, 'raw answer')
    result = state.result()
    assert result['metrics']['full_stream']['invalid_predictions'] == 1
    assert result['metrics']['full_stream']['pearson'] is None
    assert result['metrics']['common_integer_seconds']['pearson'] == pytest.approx(1)
    with pytest.raises(RuntimeError):
        state.accept(.5, 'extra step')


def test_grm_future_endpoint_is_source_endpoint_and_all_views_share_mapping():
    row = source()
    entry = fps_map(row, {i: i/180 for i in range(181)}, 10)
    ep = mapped_episode(row, entry, '/images', 'grm')
    assert ep['frames'][0]['image'][0] == '/images/00000000-0.png'
    assert ep['frames'][-1]['image'][0] == '/images/00000180-0.png'
    assert ep['frames'][10]['image'] == [f'/images/00000030-{v}.png' for v in range(3)]
    with pytest.raises(ValueError):
        mapped_episode(row, playback_map(row, 'no_memory'), '/images', 'grm')


@pytest.mark.parametrize('group,condition,rate', [(2, 'no_memory', '30'),
                                               (5, 'fps20', '30'), (5, 'fps20', '30/1')])
def test_frozen_loader_checks_receipt_plan_and_supported_conditions(tmp_path, group, condition, rate):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    row = source()
    row['source_rate'] = rate
    entry = (playback_map(row, 'no_memory') if group == 2 else
             fps_map(row, {i: i/180 for i in range(181)}, 20))
    plan = tmp_path/'plan.json'
    write(plan, {'fixed': True})
    baseline = tmp_path/'group1-manifest.json'
    write(baseline, {'episodes': [row]})
    path = tmp_path/'streams'/f'group{group}'/f'{condition}.json'
    value = {'group': group, 'condition': condition, 'plan_sha256': sha(plan),
             'baseline_manifest_sha256': sha(baseline), 'expected_episodes': 1,
             'expected_samples': len(entry['target_pt']), 'episodes': [entry]}
    write(path, value)
    write(tmp_path/'streams/preparation.json', {'manifests': {str(path): {'sha256': sha(path)}}})
    write(tmp_path/'streams/verification.json', {'status': 'passed', 'all_maps_independently_checked': True})
    loaded, rows, _ = load_inputs(group, condition, 'procvlm-2b', tmp_path, plan)
    assert loaded == value and rows[row['key']] == row
    for bad_group, bad_condition, model in [(2, '../no_memory', 'procvlm-2b'),
                                           (2, 'no_memory', 'robobrain25-4b'),
                                           (5, 'fps25', 'procvlm-2b')]:
        with pytest.raises(ValueError):
            load_inputs(bad_group, bad_condition, model, tmp_path, plan)
    changed = deepcopy(value)
    changed['episodes'][0]['target_pt'][1] += .1
    write(path, changed)
    with pytest.raises(AssertionError):
        load_inputs(group, condition, 'procvlm-2b', tmp_path, plan)
    write(path, value)
    write(plan, {'fixed': False})
    with pytest.raises(AssertionError):
        load_inputs(group, condition, 'procvlm-2b', tmp_path, plan)
