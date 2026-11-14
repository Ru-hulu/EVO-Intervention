"""Separate source-frame identity, forward playback position, and model inputs."""
from fractions import Fraction
import math
from pathlib import Path
from experiments.prm_five_groups_20261114.eval_common import uniform_indices
from experiments.soc_perturbations_20261114.verify import verify_playback


def timeline(row, entry):
    verify_playback(row, entry)
    events = {e['after_source_step']: e for e in entry['events']}
    rate = Fraction(row['source_rate'])
    original_positions, positions = [], []
    inserted = 0
    for i, frame in enumerate(row['frame_indices']):
        original_positions.append(len(positions))
        anchor = Fraction(frame) + inserted*rate/5
        positions.append(math.ceil(anchor))
        if i in events:
            event = events[i]
            for k in range(1, event['inserted_steps']+1):
                positions.append(math.ceil(anchor+k*rate/5))
            inserted += event['inserted_steps']
    assert len(positions) == len(entry['source_frame_indices'])
    assert all(a < b for a, b in zip(positions, positions[1:]))
    return dict(original_positions=original_positions, playback_positions=positions)


def model_episode(row, entry):
    """PT, events and source timestamps are excluded from the inference boundary."""
    clock = timeline(row, entry)
    root = Path(row['image_root'])
    return dict(key=row['key'], task=row['task_prompt'], positions=clock['playback_positions'],
        images=[[str(root/f'{frame:08d}-{v}.png') if row['videos'][v] else None
                 for v in range(3)] for frame in entry['source_frame_indices']])


def image_slots(episode, step, model):
    assert 0 <= step < len(episode['images'])
    if model == 'procvlm-2b':
        steps = [max(0, i) for i in range(step-7, step+1)]
        return [(i, 0) for i in steps]
    if model == 'robometer-4b':
        return [(int(i), 0) for i in uniform_indices(step)[:7]] + [(step, v) for v in range(3)]
    assert model == 'forest'
    return [(step, v) for v in range(3)]


def prediction_valid(value):
    return (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1)


def validate_record(record, row, entry, identity, complete=False):
    assert record['identity'] == identity
    n = len(record['predictions'])
    assert 0 <= n <= len(entry['target_pt'])
    if complete:
        assert n == len(entry['target_pt'])
    for key in ('source_frame_indices', 'playback_timestamps', 'target_pt'):
        assert record[key] == entry[key][:n]
    assert record['progress_sha256'] == row['progress_sha256']
    assert record['source_video_sha256'] == row['video_hashes']
    assert record['playback_positions'] == timeline(row, entry)['playback_positions'][:n]
    assert len(record['raw_responses']) == len(record['input_audits']) == n
    assert all(v is None or prediction_valid(v) for v in record['predictions'])
    assert record['memory_enabled'] is True and record['future_endpoint_reference'] is False
    episode = model_episode(row, entry)
    for i, audit in enumerate(record['input_audits']):
        assert audit['playback_step'] == i and audit['playback_position'] == episode['positions'][i]
        expected = image_slots(episode, i, identity['model'])
        assert [tuple(v) for v in audit['input_playback_slots']] == expected
        assert audit['input_images'] == len(expected)
    a = record['initialization']
    for key in ('model', 'suite', 'checkpoint_sha256'):
        assert a[key] == identity[key]
    assert a['memory_enabled'] and a['strict_final_state'] and a['epoch'] == 1
    assert not a['future_endpoint_reference'] and not a['previous_text_used']
    from .common import digest
    assert digest(a) == record['initialization_sha256']
    return n
