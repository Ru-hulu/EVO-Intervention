"""One episode, per-frame durable prefixes, vision-only memory reconstruction."""
import copy
import math
from pathlib import Path
from .common import save, read, digest
from .protocol import model_episode, timeline, validate_record, prediction_valid
from .metrics import score


class SafetyPause(Exception):
    pass


def json_raw(value):
    """Preserve a nonfinite numeric response as text rather than emit invalid JSON."""
    if isinstance(value,float) and not math.isfinite(value):return dict(nonfinite=repr(value))
    if isinstance(value,dict):return {k:json_raw(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_raw(v) for v in value]
    return value


def empty_record(row, entry, identity, audit):
    value = dict(identity=identity, predictions=[], raw_responses=[], input_audits=[],
        source_frame_indices=[], playback_timestamps=[], playback_positions=[], target_pt=[],
        progress_sha256=row['progress_sha256'], source_video_sha256=row['video_hashes'],
        memory_enabled=True, future_endpoint_reference=False, initialization=copy.deepcopy(audit),
        initialization_sha256=digest(audit), status='partial')
    return value


def run_episode(backend, row, entry, identity, destination, check=lambda: None, progress=lambda n: None):
    destination = Path(destination)
    partial = destination.with_suffix('.partial.json')
    if destination.exists():
        record = read(destination)
        validate_record(record, row, entry, identity, complete=True)
        assert record['status'] == 'completed'
        assert record['initialization'] == backend.audit
        return record
    record = read(partial) if partial.exists() else empty_record(row, entry, identity, backend.audit)
    start = validate_record(record, row, entry, identity)
    assert record['initialization'] == backend.audit, 'Initialization changed during resume'
    episode = model_episode(row, entry)
    backend.begin(episode)
    # ProcVLM/Robometer reconstruct bounded windows directly from the prefix;
    # FoREST reconstructs vision memory without invoking its language model.
    for i in range(start):
        check(); backend.rebuild_step(i)
    for i in range(start, len(entry['target_pt'])):
        check()
        output = backend.predict(i)
        value, raw = output['prediction'], output['raw_response']
        if value is not None and not prediction_valid(value):
            raw = dict(raw_response=raw, invalid_prediction=repr(value)); value = None
        record['predictions'].append(value); record['raw_responses'].append(json_raw(raw))
        record['input_audits'].append(output['input_audit'])
        for k in ('source_frame_indices', 'playback_timestamps', 'target_pt'):
            record[k].append(entry[k][i])
        record['playback_positions'].append(episode['positions'][i])
        save(partial, record)
        progress(i+1)
    validate_record(record, row, entry, identity, complete=True)
    record.update(status='completed', metrics=score(record['predictions'], row, entry))
    save(destination, record)
    # Keep the durable prefix too, as an audit of the unmodified inference output.
    return record


def resume_canary(backend, row, entry, identity, destination, check=lambda: None):
    """Real complete trajectory + interrupted replay inside both perturbations.

    These predictions are isolated under canary/, never formal experiment data.
    Continuous and resumed calls use singleton inference and the same weights.
    """
    destination = Path(destination)
    reference = run_episode(backend, row, entry, identity, destination/'continuous.json', check)
    comparisons = []
    for j, event in enumerate(entry['events']):
        cut = event['first_inserted_step'] + max(1, event['duration_steps']//2)
        prefix = copy.deepcopy(reference)
        for k in ('predictions', 'raw_responses', 'input_audits', 'source_frame_indices',
                  'playback_timestamps', 'playback_positions', 'target_pt'):
            prefix[k] = prefix[k][:cut]
        prefix.pop('metrics', None); prefix['status'] = 'partial'
        path = destination/f'resume-event{j}.json'
        if not path.exists():
            save(path.with_suffix('.partial.json'), prefix)
        resumed = run_episode(backend, row, entry, identity, path, check)
        assert resumed['predictions'][:cut] == reference['predictions'][:cut]
        assert resumed['raw_responses'][:cut] == reference['raw_responses'][:cut]
        differences = []
        for a,b in zip(reference['predictions'][cut:], resumed['predictions'][cut:]):
            assert (a is None) == (b is None), 'Resume changed invalid-answer positions'
            if a is not None:
                differences.append(abs(a-b))
        error = max(differences, default=0)
        assert error <= .002, f'Resume differs from continuous native inference: {error}'
        comparisons.append(dict(event=j, prefix_observations=cut, max_abs_difference=error,
            compared_finite_predictions=len(differences), invalid_predictions=sum(v is None for v in resumed['predictions'][cut:])))
    assert any(prediction_valid(p) for p in reference['predictions']), 'Canary produced no valid predictions'
    return comparisons
