"""Recover metrics with explicit annotation eligibility and native FPL rules."""
import ast
from functools import lru_cache
import math
from pathlib import Path
from typing import Sequence

from .prepare_cache import sha


@lru_cache(None)
def fpl_functions():
    # Execute only the unmodified pure functions from the pinned vendored source;
    # avoid importing the benchmark's unrelated dataset/model dependencies.
    path = Path(__file__).parent/'metric_reference/fpl.py'
    names = {'_anchor_frames', '_maximum_drawdown', '_zigzag_descents', '_select_descent'}
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = dict(Sequence=Sequence, Episode=object, VMBMKError=ValueError,
                     FPL_SAMPLE_RATE_HZ=5.)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, sha(path)


def validate_events(events, row, native_truth):
    assert events['schema'] == 'prm-recover-events-v1'
    assert events['review_status'] == 'verified'
    assert events['key'] == row['key'] and events['video_sha256'] == row['video_hashes'][0]
    assert events['progress_sha256'] == row['progress_sha256']
    assert events['total_frames'] == row['frames']
    assert events.get('success') is None or type(events['success']) is bool
    onset = events.get('failure_onset_frame')
    assert onset is None or (type(onset) is int and 0 <= onset < row['frames'])
    assert events.get('trr_role') in (None, 'normal', 'r_positive', 'r_zero')
    occupied = set()
    for segment in events.get('abnormal_plateaus', []):
        assert segment['cause'] in ('repeated_action', 'wrong_action')
        start, end = segment['start_frame'], segment['end_frame']
        assert type(start) is type(end) is int and 0 <= start <= end < row['frames']
        frames = set(range(start, end+1))
        assert not frames & occupied, 'Overlapping abnormal intervals'
        occupied |= frames
        assert all(native_truth[i] == native_truth[start] for i in frames), 'Annotated interval is not flat'
    evidence = events['annotation_provenance']
    assert evidence and all(sha(x['path']) == x['sha256'] for x in evidence)
    return events


def score_recover(pred, truth, indices, timestamps, total_frames, events=None):
    assert len(pred) == len(truth) == len(indices) == len(timestamps) > 0
    valid = lambda p: p is not None and math.isfinite(p)
    result = {'successful_endpoint_score': None, 'fpl': None,
              'abnormal_plateau_absolute_mae': None, 'abnormal_plateau_drift_mae': None,
              'metric_reasons': {}, 'plateau_segments': [], 'units': 'unit PT (0 to 1)',
              'plateau_aggregation': 'frame weighted within episode; equal episode weight between episodes'}
    if events is None:
        result['metric_reasons'] = {k: 'missing_verified_event_annotation' for k in (
            'successful_endpoint_score', 'fpl', 'abnormal_plateau_absolute_mae', 'abnormal_plateau_drift_mae')}
        return result
    reasons = result['metric_reasons']
    if events.get('success') is True:
        cutoff = timestamps[-1] - .02*(timestamps[-1]-timestamps[0])
        selected = [i for i, t in enumerate(timestamps) if t >= cutoff]
        result['endpoint_scored_indices'] = selected
        if all(valid(pred[i]) for i in selected):
            result['successful_endpoint_score'] = math.fsum(pred[i] for i in selected)/len(selected)
        else:
            reasons['successful_endpoint_score'] = 'invalid_endpoint_predictions'
    else:
        reasons['successful_endpoint_score'] = ('not_labeled_success' if events.get('success') is False
                                               else 'missing_success_label')
    onset = events.get('failure_onset_frame')
    if events.get('trr_role') == 'r_zero':
        reasons['fpl'] = 'official_r_zero_exclusion'
    elif events.get('trr_role') == 'normal' and onset is None:
        reasons['fpl'] = 'no_annotated_failure'
    elif onset is None or events.get('trr_role') is None:
        reasons['fpl'] = 'missing_explicit_failure_onset_or_role'
    elif not all(valid(p) for p in pred):
        reasons['fpl'] = 'invalid_predictions'
    elif total_frames < 2:
        reasons['fpl'] = 'zero_duration'
    else:
        native, source_sha = fpl_functions()
        curve = list(zip(indices, pred))
        drop = native['_maximum_drawdown'](curve)
        descents = native['_zigzag_descents'](curve, .2*drop) if drop > 0 else []
        chosen = native['_select_descent'](descents)
        last = total_frames-1
        error = abs(chosen[0]-onset) if chosen else max(onset, last-onset)
        result['fpl'] = error/last
        result['fpl_details'] = dict(ground_truth_onset_frame=onset,
            predicted_onset_frame=chosen[0] if chosen else None, normalized_error=error/last,
            no_descent_penalty=chosen is None, maximum_drawdown=drop, zigzag_threshold=.2*drop,
            zigzag_descents=descents, selected_descent=chosen, reference_sha256=source_sha)
    absolute, drift = [], []
    for interval in events.get('abnormal_plateaus', []):
        start, end = interval['start_frame'], interval['end_frame']
        selected = [i for i, frame in enumerate(indices) if start <= frame <= end]
        segment = dict(interval, scored_indices=selected, absolute_mae=None, drift_mae=None)
        if not selected:
            segment['reason'] = 'interval_has_no_observation'
        elif not all(valid(pred[i]) for i in selected):
            segment['reason'] = 'invalid_predictions'
        else:
            first = selected[0]
            segment['scored_anchor_frame'] = indices[first]
            segment['anchor_rule'] = 'first observed frame inside annotated interval; no interpolation'
            a = [abs(pred[i]-truth[i]) for i in selected]
            d = [abs((pred[i]-pred[first])-(truth[i]-truth[first])) for i in selected]
            absolute += a; drift += d
            segment.update(absolute_mae=math.fsum(a)/len(a), drift_mae=math.fsum(d)/len(d))
        result['plateau_segments'].append(segment)
    segments = result['plateau_segments']
    if segments and all(s['absolute_mae'] is not None for s in segments):
        result['abnormal_plateau_absolute_mae'] = math.fsum(absolute)/len(absolute)
        result['abnormal_plateau_drift_mae'] = math.fsum(drift)/len(drift)
    else:
        reason = 'no_annotated_abnormal_plateau' if not segments else 'one_or_more_plateau_segments_undefined'
        for key in ('abnormal_plateau_absolute_mae', 'abnormal_plateau_drift_mae'):
            reasons[key] = reason
    return result
