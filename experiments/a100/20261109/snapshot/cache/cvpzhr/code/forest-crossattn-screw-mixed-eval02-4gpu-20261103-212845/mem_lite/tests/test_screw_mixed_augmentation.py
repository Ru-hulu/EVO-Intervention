"""The augmentation must keep source supervision and complete a closed replay loop."""

from collections import Counter

import pytest

from mem_lite.datasets.prepare_screw_mixed import (
    choose_events, combine_rounds, expand_schedule, sample_indices,
)


def test_three_round_labels_keep_rollback_and_idle():
    rounds = [(0, 4, [0, 0.9, 0.3, 1]), (6, 8, [0, 1]), (8, 11, [0, 0.6, 1])]
    assert combine_rounds(rounds) == pytest.approx(
        [0, 0.3, 0.1, 1/3, 1/3, 1/3, 1/3, 2/3, 2/3, 2.6/3, 1])


def test_jitter_reverses_each_frame_then_returns_before_resuming():
    events = [dict(kind='jitter', anchor=3, frames=2)]
    rows = list(expand_schedule([0, 6, 12, 18, 24], 30, events))
    assert [r['source_sample_index'] for r in rows] == [0, 1, 2, 3, 2, 1, 2, 3, 4]
    assert [r['frame_index'] for r in rows] == list(range(0, 54, 6))
    assert [r['kind'] for r in rows[4:8]] == ['reverse', 'reverse', 'forward', 'forward']
    pt = [0, 0.8, 0.2, 0.6, 1]
    assert [pt[r['source_sample_index']] for r in rows] == [0, 0.8, 0.2, 0.6, 0.2, 0.8, 0.2, 0.6, 1]


def test_hold_then_loop_preserves_all_originals_and_the_short_final_interval():
    indices = [0, 6, 12, 13]
    events = [dict(kind='jitter', anchor=2, frames=2), dict(kind='stationary', anchor=2, frames=3)]
    rows = list(expand_schedule(indices, 30, events))
    assert [r['source_sample_index'] for r in rows] == [0, 1, 2, 2, 2, 2, 1, 0, 1, 2, 3]
    assert [r['frame_index'] for r in rows] == [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 55]
    assert [r['source_sample_index'] for r in rows if r['kind']=='original'] == [0, 1, 2, 3]


@pytest.mark.parametrize('frames,anchors', [(360,1), (450,1), (451,2), (900,2), (901,3), (1800,4)])
def test_anchor_count_is_exact_duration_ceiling_per_augmentation(frames, anchors):
    indices = sample_indices(frames, 30)
    events = choose_events(indices, frames, 30, 42, 'example')
    assert Counter(e['kind'] for e in events) == {'stationary':anchors, 'jitter':anchors}
    assert events == choose_events(indices, frames, 30, 42, 'example')
    for kind in ('stationary', 'jitter'):
        selected = [e for e in events if e['kind']==kind]
        assert len({e['anchor'] for e in selected}) == anchors
        assert all(0<=e['frames']<=25 if kind=='stationary' else 0<=e['frames']<=min(10,e['anchor']) for e in selected)


def test_validation_has_no_augmentation_and_last_source_frame_is_retained():
    indices = sample_indices(32, 30)
    assert indices == [0, 6, 12, 18, 24, 30, 31]
    assert choose_events(indices, 32, 30, 42, 'validation', enabled=False) == []
    rows = list(expand_schedule(indices, 30, []))
    assert [r['frame_index'] for r in rows] == indices


def test_zero_length_events_and_episode_start_do_not_add_fake_observations():
    events = [dict(kind='stationary', anchor=0, frames=0), dict(kind='jitter', anchor=0, frames=0)]
    rows = list(expand_schedule([0, 6], 30, events))
    assert [r['source_sample_index'] for r in rows] == [0, 1]


def test_overlapping_rounds_are_rejected():
    with pytest.raises(ValueError, match='Overlapping'):
        combine_rounds([(0,2,[0,1]), (1,3,[0,1])])
