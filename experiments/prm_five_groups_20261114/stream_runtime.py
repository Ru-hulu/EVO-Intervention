"""CPU protocol adapter for official group-2/5 playback and exact prefix resume.

This module does not load a model, decode media, launch GPU work or change the
baseline dispatcher. A future worker must supply audited native engines and
verified image paths. Robometer/FoREST use their separate tensor/memory adapters.
"""
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

from .eval_common import EVAL, PLAN, trajectory_metrics
from .official_baseline import KINDS, next_history, vlac_next
from .prepare_cache import sha
from .stream_manifests import digest, score_fps_stream


def validate_entry(row, entry):
    data = dict(entry)
    expected = data.pop('map_sha256')
    assert digest(data) == expected, 'Playback map changed'
    for key in ('key', 'suite', 'task_id', 'progress_sha256'):
        assert entry[key] == row[key], key
    assert entry['input_issues'] == row.get('input_issues', [])
    frames, times, truth = (entry[k] for k in (
        'source_frame_indices', 'playback_timestamps', 'target_pt'))
    assert len(frames) == len(times) == len(truth) > 0
    assert all(type(f) is int and 0 <= f < row['frames'] for f in frames)
    assert all(math.isfinite(t) for t in times + truth)
    assert all(a < b for a, b in zip(times, times[1:]))
    assert entry['all_views_share_source_frame_map'] is True
    if 'source_step_indices' in entry:
        steps = entry['source_step_indices']
        assert all(type(i) is int and 0 <= i < len(row['frame_indices']) for i in steps)
        assert frames == [row['frame_indices'][i] for i in steps]
        assert truth == [row['target_pt'][i] for i in steps]
    return entry


def load_inputs(group, condition, model, evaluation=EVAL, plan=PLAN):
    """Require a frozen manifest plus the independent full-map verification."""
    if model not in KINDS or group not in (2, 5):
        raise ValueError('This adapter supports the five official native engines only')
    allowed = ({'no_memory'} | {f'{v}-seed{s}' for v in ('static_only', 'rewind_only')
                              for s in (42, 43, 44)}) if group == 2 else {
                                  f'fps{f}' for f in (30, 20, 10, 5, 2)}
    if condition not in allowed or (group == 2 and model not in ('procvlm-2b', 'r2vlm-alfred-rl')):
        raise ValueError('Unconfirmed model/condition for this adapter')
    evaluation = Path(evaluation)
    path = evaluation/'streams'/f'group{group}'/f'{condition}.json'
    preparation = json.loads((evaluation/'streams/preparation.json').read_text())
    verified = json.loads((evaluation/'streams/verification.json').read_text())
    assert verified['status'] == 'passed' and verified['all_maps_independently_checked'] is True
    assert sha(path) == preparation['manifests'][str(path)]['sha256']
    manifest = json.loads(path.read_text())
    baseline = evaluation/'group1-manifest.json'
    assert manifest['group'] == group and manifest['condition'] == condition
    assert manifest['baseline_manifest_sha256'] == sha(baseline)
    assert manifest['plan_sha256'] == sha(plan)
    sources = {r['key']: r for r in json.loads(baseline.read_text())['episodes']}
    entries = manifest['episodes']
    assert len(entries) == len({e['key'] for e in entries}) == manifest['expected_episodes']
    assert sum(len(e['source_frame_indices']) for e in entries) == manifest['expected_samples']
    for entry in entries:
        validate_entry(sources[entry['key']], entry)
        if group == 2:
            assert entry['input_fps'] == 5
            assert entry['memory_enabled'] is (condition != 'no_memory')
            expected_condition = condition.split('-seed', 1)[0]
            assert entry['condition'] == expected_condition
            expected_seed = int(condition.split('-seed')[1]) if '-seed' in condition else None
            assert entry['seed'] == expected_seed
        else:
            assert entry['condition'] == condition and entry['input_fps'] == int(condition[3:])
            assert entry['memory_enabled'] is True and entry['source_rate'] == '30'
            assert Fraction(sources[entry['key']]['source_rate']) == 30
    return manifest, sources, sha(path)


def mapped_episode(row, entry, image_root, kind):
    """Only images, instruction and a memory-mode flag cross the model boundary."""
    validate_entry(row, entry)
    if kind not in ('grm', 'proc', 'r2', 'vlac'):
        raise ValueError('Unsupported native engine')
    if not entry['memory_enabled'] and kind not in ('proc', 'r2'):
        raise ValueError('No confirmed no-memory condition for this engine')
    root = Path(image_root)
    return {'task': row['task_prompt'], '_memory_enabled': entry['memory_enabled'],
            'frames': [{'image': [str(root/f'{f:08d}-{view}.png')
                        if path and (view == 0 or kind == 'grm') else None
                        for view, path in enumerate(row['videos'])]}
                       for f in entry['source_frame_indices']]}


def install_proc_memory_policy(native):
    """Use one actual image for ProcVLM ablation; keep its native text/parser.

    Apply only to the new worker's independently loaded native module. Baseline
    processes and other engines keep their existing image interface unchanged.
    """
    original = native.images_for

    def images_for(kind, episode, index):
        if kind == 'proc' and episode.get('_memory_enabled') is False:
            return [episode['frames'][index]['image'][0]]
        return original(kind, episode, index)

    native.images_for = images_for


class StreamSession:
    """Sequential playback; memory follows presented steps, including rewinds."""
    def __init__(self, kind, episode, entry, identity, initial_history, prefix=None):
        if kind not in ('grm', 'proc', 'r2', 'vlac'):
            raise ValueError('Unsupported native engine')
        for key in ('model', 'official_revision', 'plan_sha256', 'manifest_sha256'):
            assert identity.get(key), key
        assert KINDS[identity['model']] == kind
        data = dict(entry)
        assert data.pop('map_sha256') == digest(data), 'Playback map changed'
        assert len(episode['frames']) == len(entry['source_frame_indices'])
        assert episode['_memory_enabled'] is entry['memory_enabled']
        self.kind, self.episode, self.entry = kind, deepcopy(episode), deepcopy(entry)
        self.identity = dict(identity, map_sha256=entry['map_sha256'], key=entry['key'],
                             memory_enabled=entry['memory_enabled'])
        self.initial_history = initial_history
        self.predictions, self.raw = ([0.], [None]) if kind == 'vlac' else ([], [])
        if prefix is not None:
            assert prefix['identity'] == self.identity, 'Resume identity mismatch'
            n = len(prefix['predictions'])
            assert n <= len(entry['source_frame_indices']) and len(prefix['raw_responses']) == n
            for field in ('source_frame_indices', 'playback_timestamps', 'target_pt'):
                assert prefix[field] == entry[field][:n], field
            assert all(v is None or (type(v) in (int, float) and math.isfinite(v))
                       for v in prefix['predictions'])
            if kind == 'vlac':
                assert n >= 1 and prefix['predictions'][0] == 0. and prefix['raw_responses'][0] is None
            assert all(isinstance(x, str) for x in prefix['raw_responses'][1 if kind == 'vlac' else 0:])
            self.predictions = list(prefix['predictions'])
            self.raw = list(prefix['raw_responses'])

    @property
    def complete(self):
        return len(self.predictions) == len(self.entry['source_frame_indices'])

    def request(self):
        if self.complete:
            raise RuntimeError('Stream is already complete')
        history = self.initial_history
        if self.kind == 'r2' and self.entry['memory_enabled'] and self.raw:
            history = next_history(self.raw[-1])
        return self.episode, len(self.predictions), history

    def accept(self, value, raw):
        if self.complete:
            raise RuntimeError('Cannot extend a completed stream')
        assert isinstance(raw, str)
        if value is not None:
            assert type(value) in (int, float)
            value = float(value) if math.isfinite(value) else None
        if self.kind == 'vlac':
            value = vlac_next(self.predictions[-1], value)
        self.predictions.append(value)
        self.raw.append(raw)

    def checkpoint(self):
        n = len(self.predictions)
        return {'identity': dict(self.identity), 'predictions': list(self.predictions),
                'raw_responses': list(self.raw),
                **{field: self.entry[field][:n] for field in (
                    'source_frame_indices', 'playback_timestamps', 'target_pt')}}

    def result(self):
        if not self.complete:
            raise RuntimeError('Partial streams cannot become final results')
        scores = (score_fps_stream(self.predictions, self.entry)
                  if 'common_scoring_indices' in self.entry else
                  {'full_stream': trajectory_metrics(self.predictions, self.entry['target_pt'],
                                                     self.entry['playback_timestamps'])})
        return {**self.checkpoint(), 'stream_map': deepcopy(self.entry), 'metrics': scores}
