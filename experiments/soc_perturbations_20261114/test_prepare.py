from copy import deepcopy
import pytest

from .prepare import choose, EXCLUDED
from .verify import verify_playback
from experiments.prm_five_groups_20261114.stream_manifests import playback_map, digest


def row():
    return dict(key='State/S01/test', suite='State', task_id='S01', progress_sha256='test',
        frames=241, frame_indices=list(range(0, 241, 6)), timestamps=[i/5 for i in range(41)],
        target_pt=[(i/40)**2 for i in range(41)], input_issues=[])


def test_selection_has_two_validation_only_and_excludes_short_view():
    tasks = {s: [dict(task_id='O03' if s == 'Order' else 'X01', train=['train'],
        val=['a','b','c','d']+(['swap-blocks_episode_001342'] if s == 'Order' else []))]
        for s in ('State', 'Order', 'Count')}
    a = choose(tasks)
    assert len(a) == 6 and a == choose(tasks)
    assert all(k != 'train' and f'{s}/{t}/{k}' != EXCLUDED for s, t, k in a)


@pytest.mark.parametrize('variant', ['static_only', 'rewind_only'])
@pytest.mark.parametrize('seed', [42, 43, 44])
def test_independent_playback_and_nonlinear_pt(variant, seed):
    source = row(); entry = playback_map(source, variant, seed)
    verify_playback(source, entry)
    for event in entry['events']:
        first = event['first_inserted_step']; last = event['last_inserted_step']
        if variant == 'static_only':
            assert len(set(entry['target_pt'][first-1:last+1])) == 1
        else:
            assert entry['target_pt'][first] < entry['target_pt'][first-1]
            assert entry['target_pt'][last] == entry['target_pt'][first-1]


def test_rehashed_wrong_truth_is_rejected():
    entry = playback_map(row(), 'rewind_only', 42)
    entry['target_pt'][entry['events'][0]['first_inserted_step']] += .01
    entry['map_sha256'] = digest({k:v for k,v in entry.items() if k != 'map_sha256'})
    with pytest.raises(AssertionError): verify_playback(row(), entry)


def test_rehashed_dropped_original_is_rejected():
    entry = playback_map(row(), 'static_only', 42)
    for key in ('source_step_indices','source_frame_indices','source_timestamps','playback_timestamps','target_pt'):
        del entry[key][3]
    entry['map_sha256'] = digest({k:v for k,v in entry.items() if k != 'map_sha256'})
    with pytest.raises(AssertionError): verify_playback(row(), entry)
