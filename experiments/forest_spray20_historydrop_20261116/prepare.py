"""Freeze replacement draws, independent dropout, and source bindings on CPU."""
import argparse
import copy
import math
import random
import time
from .common import *


def failure_windows(annotation, margin_seconds):
    if margin_seconds < 0:
        raise ValueError('Negative margin')
    fps, end = annotation['fps'], annotation['total_frames'] - 1
    windows = []
    for stage in annotation['stages']:
        if not stage.get('outcome', '').startswith('failed'):
            continue
        start, stop = stage['source_start_frame'], stage['source_end_frame']
        if not 0 <= start <= stop <= end:
            raise ValueError('Invalid failure annotation interval')
        windows.append(dict(start=max(0, math.ceil(start-margin_seconds*fps)),
            end=min(end, math.floor(stop+margin_seconds*fps)),
            failed_start=start, failed_end=stop, outcome=stage['outcome']))
    if not windows:
        raise ValueError('No annotated failure; do not invent a PT-derived event')
    return windows


def make_draws(samples, epochs, world_size, seed, multiplier, dropout):
    if not samples or multiplier <= 1 or not 0 <= dropout <= 1 or world_size < 1:
        raise ValueError('Invalid sampling configuration')
    total = math.ceil(len(samples)/world_size)*world_size
    weights = [multiplier if s['near_anomaly'] else 1 for s in samples]
    result = []
    for epoch in range(epochs):
        indices = random.Random(seed+epoch).choices(range(len(samples)), weights=weights, k=total)
        memory_rng = random.Random(seed+1_000_003+epoch)
        draws = [dict(index=i, drop_history=memory_rng.random() < dropout) for i in indices]
        result.append(dict(epoch=epoch, draws=draws, near_anomaly_draws=sum(samples[i]['near_anomaly'] for i in indices),
            history_dropout_draws=sum(d['drop_history'] for d in draws), unique_frames=len(set(indices)),
            episodes_visited=len({samples[i]['episode'] for i in indices})))
    return total, result


def main(margin=2.0, multiplier=5.0):
    verify_code()
    assert not (ROOT/'plan.json').exists(), 'Refuse replacing a frozen training plan'
    assert read(CACHE_SETUP/'final-verification.json')['status'] == 'passed'
    selection_path = CACHE_SETUP/'resolved-selection.json'; selection = read(selection_path)
    selected = set(selection['selection']['R03'])
    assert len(selected) == 20 and not selected.intersection(selection['excluded_validation'])
    ready = read(CACHE/'cache-ready.json'); native = copy.deepcopy(read(OLD/'forest/config.json')['native'])
    assert ready['samples'] == 6002 and ready['episodes'] == 20
    assert ready['selection_sha256'] == sha(selection_path)
    assert native['history_source'] == ready['history_source']
    assert native['sample_files'] == ready['sample_files']
    bindings = {}
    def bind(path, digest=None):
        actual = sha(path)
        if digest is not None: assert actual == digest, path
        bindings[str(path)] = actual
        return actual
    for path in (selection_path, CACHE/'cache-ready.json', CACHE_SETUP/'final-verification.json', OLD/'forest/config.json'):
        bind(path)
    for path, digest in selection['source_sha256'].items(): bind(path, digest)
    by_episode = {r['sample_id']:r for r in selection['records']}
    samples, events = [], {}
    for directory in map(Path, ready['sample_files']):
        r = by_episode[directory.name]
        assert r['split'] == 'train' and r['task_id'] == 'R03'
        ann = read(r['annotation_path']); bind(r['annotation_path'], r['annotation_sha256'])
        windows = failure_windows(ann, margin); events[directory.name] = windows
        assert ann['total_frames'] == r['frames'] and ann['fps'] == r['fps']
        bind(directory/'samples.jsonl'); bind(directory/'summary.json')
        for row in rows(directory/'samples.jsonl'):
            frame = row['frame_index']
            samples.append(dict(episode=directory.name, frame_index=frame,
                near_anomaly=any(w['start'] <= frame <= w['end'] for w in windows)))
    assert len(samples) == 6002
    total, epochs = make_draws(samples, 2, 4, 42, multiplier, .5)
    protocol = dict(anomaly_definition='annotated failed operation interval plus margin; not GT decline or model prediction',
        margin_seconds=margin, per_frame_weight_multiplier=multiplier, replacement=True,
        history_dropout_probability=.5, history_dropout_unit='independent Bernoulli per draw, before collation',
        history_dropout_independent_of_anomaly=True, seed=42, base_samples_per_epoch=6002,
        draws_per_epoch=total, extra_draws_for_equal_four_rank_lengths=total-6002,
        input_fps=5, history_source_unchanged=True, current_views_and_targets_unchanged=True)
    plan = dict(format='forest-training-draws-v1', data_signature=ready['data_signature'], world_size=4,
        draws_per_epoch=total, samples=samples, epochs=epochs, protocol=protocol, failure_windows=events)
    save(ROOT/'training-draws.json', plan); bind(ROOT/'training-draws.json')
    native.update(output_dir=str(OUT), epochs=2, batch_size=2, gradient_accumulation_steps=1,
        max_steps=None, resource_stop_path=str(ROOT/'resource-stop.json'),
        training_draws=dict(path=str(ROOT/'training-draws.json'), sha256=sha(ROOT/'training-draws.json')))
    native['run_protocol'].update(world_size=4, gpus=[0,1,2,3], epochs=2, effective_batch=8,
        history_dropout_probability=.5, anomaly_sampling=protocol, old_training_weights_reused=False)
    assert native['learning_rate'] == 1e-4 and native['lr_schedule'] == 'constant'
    assert native['data_signature'] == ready['data_signature']
    receipt_path = Path('/cache/cvpzhr/setup/qwen35-original-download-20261113/verification.json')
    receipt = read(receipt_path); bind(receipt_path)
    for name, item in receipt['files'].items(): bind(Path(native['model_path'])/name, item['sha256'])
    save(ROOT/'training.config.json', native); bind(ROOT/'training.config.json')
    audit = dict(status='passed', expected_episodes=20, original_samples=6002,
        original_anomaly_frames=sum(s['near_anomaly'] for s in samples), protocol=protocol,
        expected_anomaly_probability=sum(s['near_anomaly']*multiplier for s in samples)/sum(multiplier if s['near_anomaly'] else 1 for s in samples),
        epochs=[{k:v for k,v in e.items() if k != 'draws'} for e in epochs],
        selection_sha256=sha(selection_path), data_signature=ready['data_signature'],
        no_validation_trajectories=True, annotation_source='Existing visual stage annotations; human_verified=false',
        created=time.time())
    save(ROOT/'sampling-verification.json', audit)
    save(ROOT/'plan.json', dict(status='prepared', bindings=bindings, expected_steps=2*math.ceil(total/8),
        four_gpu_single_job=True, fresh_initialization=True, evaluation_enabled=False, sampling=audit))
    print(audit, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--margin', type=float, default=2); p.add_argument('--multiplier', type=float, default=5)
    a = p.parse_args(); main(a.margin, a.multiplier)
