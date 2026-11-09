"""Prepare normal/recovery flipcup caches for an original Forest checkpoint."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time

from mem_lite.datasets.prepare_recovery import probe_video, sha256
from mem_lite.datasets.prepare_screw_mixed import (
    assigned, build_cache, choose_events, digest, read, sample_indices, verify_episode,
)

FORMAT = 'flipcup-original-forest-cache-v1'
NORMAL_TASK = 'Order/13-flip_cups_left_to_right'
RECOVERY_TASK = 'Order/15-flip_two_cups_left_to_right_with_recovery'
VIEWS = ('main_view', 'left_view', 'right_view')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def get_index(cache):
    index = read(Path(cache) / 'index.json')
    assert index['format'] == FORMAT
    assert index['index_sha256'] == digest({k: v for k, v in index.items() if k != 'index_sha256'})
    return index


def prepare(plan_path):
    import torch

    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    cache = Path(plan['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / 'index.json').exists():
        index = get_index(cache)
        assert index['plan_sha256'] == sha256(plan_path)
        return index
    checkpoint = Path(plan['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert saved['format'] == 'mem-lite-streamforest-v1'
    assert saved['epoch'] == 1 and saved['next_batch'] == 0
    assert set(saved['model']) == {'m3_lora', 'history_projection', 'progress_head'}
    reference = saved['config']
    assert not reference['model'].get('cross_attention_layers')
    assert reference['history_source']['sample_fps'] == 5
    checkpoint_info = dict(path=str(checkpoint), sha256=sha256(checkpoint),
                           format=saved['format'], epoch=saved['epoch'], step=saved['step'])
    write(cache / 'reference-checkpoint.json', checkpoint_info)
    write(cache / 'reference-config.json', reference)
    del saved

    normal_root = Path(plan['normal_root'])
    normal_labels = Path(plan['normal_labels'])
    recovery_root = Path(plan['recovery_root'])
    prior = Path(plan['normal_base_cache'])
    split_ids = {}
    for split, filename in [('train', 'train.sample_files.json'), ('validation', 'validation.sample_files.json')]:
        paths = [Path(p) for p in read(prior / filename) if NORMAL_TASK in str(p)]
        split_ids[split] = {p.name for p in paths}
        assert all(read(p / 'summary.json')['history_source'] == reference['history_source'] for p in paths)
    assert len(split_ids['train']) == 25 and len(split_ids['validation']) == 7
    assert not split_ids['train'] & split_ids['validation']
    assert split_ids['train'] | split_ids['validation'] == {f'episode_{i:06d}' for i in range(32)}
    seen_normal = {Path(p).name for p in reference['sample_files'] if NORMAL_TASK in str(p)}
    assert seen_normal == split_ids['train'], 'Unexpected checkpoint normal-data training history'
    assert not any(RECOVERY_TASK in str(p) or 'flipcup-recovery' in str(p) for p in reference['sample_files'])
    recovery_ids = [f'episode_{i:06d}' for i in range(35)]
    recovery_validation = set(random.Random(plan['seed']).sample(recovery_ids, 7))

    records, sources = [], {str(plan_path): sha256(plan_path), str(checkpoint): checkpoint_info['sha256']}
    for kind, count, root, task in [('normal', 32, normal_root, NORMAL_TASK),
                                    ('recovery', 35, recovery_root, RECOVERY_TASK)]:
        for i in range(count):
            name = f'episode_{i:06d}'
            if kind == 'normal':
                split = 'validation' if name in split_ids['validation'] else 'train'
                metadata_path = normal_labels / name / 'metadata.json'
                progress_path = normal_labels / name / 'progress.csv'
                metadata = read(metadata_path)
                task_prompt = metadata['source']['task_prompt']
                expected_hash = metadata['units'][0]['progress_sha256']
                method = metadata['source']['pt_method']
                expected_video_hashes = metadata['source']['video_sha256']
            else:
                split = 'validation' if name in recovery_validation else 'train'
                metadata_path = root / 'progress_13stages' / f'{name}.metadata.json'
                progress_path = root / 'progress_13stages' / f'{name}.progress.csv'
                metadata = read(metadata_path)
                assert metadata['stage_count'] == 13
                annotation = root / 'videos/annotation' / f'{name}.json'
                assert sha256(annotation) == metadata['annotation_sha256']
                sources[str(annotation)] = metadata['annotation_sha256']
                expected_hash = metadata['progress_csv_sha256']
                task_prompt = plan['recovery_prompt']
                method = metadata['format_version']
                expected_video_hashes = [metadata['video_sha256'], None, None]
            assert sha256(progress_path) == expected_hash
            sources[str(progress_path)] = expected_hash
            sources[str(metadata_path)] = sha256(metadata_path)
            videos = [root / 'videos' / view / f'{name}.mp4' for view in VIEWS]
            info = [probe_video(path) for path in videos]
            assert all(item == info[0] for item in info), videos
            frames, fps = info[0]['frames'], info[0]['fps']
            assert fps == 30 and info[0]['start_time'] == 0
            for path, expected in zip(videos, expected_video_hashes, strict=True):
                actual = sha256(path)
                assert expected is None or expected == actual, str(path)
                sources[str(path)] = actual
            with progress_path.open() as stream:
                labels = list(csv.DictReader(stream))
            assert [int(row['frame_index']) for row in labels] == list(range(frames))
            pt = [float(row['pt']) for row in labels]
            assert all(math.isfinite(p) and 0 <= p <= 1 for p in pt)
            assert pt[0] == 0 and math.isclose(pt[-1], 1)
            indices = sample_indices(frames, fps)
            episode_id = f'flipcup-{kind}/{name}'
            events = choose_events(indices, frames, fps, plan['seed'], episode_id, split == 'train')
            added = sum(e['frames'] * (2 if e['kind'] == 'jitter' else 1) for e in events)
            records.append(dict(episode_id=episode_id, task=task, dataset_kind=kind,
                                source_video_id=name, source_episodes=[0], split=split,
                                videos=[str(path) for path in videos], source_frame_range=[0, frames],
                                fps=fps, duration_seconds=frames/fps, task_prompt=task_prompt,
                                progress_csv=str(progress_path), progress_sha256=expected_hash,
                                progress_method=method, original_samples=len(indices),
                                augmented_samples=len(indices)+added, events=events,
                                output_dir=str(cache / split / episode_id)))
    counts = Counter(row['split'] for row in records)
    assert counts == {'train': 53, 'validation': 14}
    index = dict(format=FORMAT, plan_path=str(plan_path), plan_sha256=sha256(plan_path),
                 checkpoint=checkpoint_info, model_path=reference['model_path'],
                 history_source=reference['history_source'], history_dim=reference['model']['history_dim'],
                 history_capacity=reference['history_capacity'], sources=sources, records=records)
    index['index_sha256'] = digest(index)
    write(cache / 'index.json', index)
    write(cache / 'split.json', {kind: {split: [r['source_video_id'] for r in records
           if r['dataset_kind'] == kind and r['split'] == split] for split in ('train', 'validation')}
           for kind in ('normal', 'recovery')})
    summary = dict(episode_counts=dict(counts),
                   original_observations={s: sum(r['original_samples'] for r in records if r['split'] == s) for s in counts},
                   observations={s: sum(r['augmented_samples'] for r in records if r['split'] == s) for s in counts},
                   anchors=dict(Counter(e['kind'] for r in records for e in r['events'])),
                   checkpoint=checkpoint_info, checkpoint_training_overlap_with_validation=False,
                   augmentation=plan['augmentation'], training_started=False)
    write(cache / 'index-summary.json', summary)
    print(json.dumps(summary), flush=True)
    return index


def worker(cache_dir, worker_id, workers):
    import torch
    from mem_lite.memory.vision import load_vision_encoder

    torch.set_num_threads(2)
    cache = Path(cache_dir)
    lock = (cache / f'worker-{worker_id}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index = get_index(cache)
    group = assigned(index, workers)[worker_id]
    state = dict(pid=os.getpid(), phase='loading_vision', completed=0, samples=0, total=len(group))
    status = cache / f'worker-{worker_id}.json'
    write(status, state)
    try:
        visual, processor, vision_hash = load_vision_encoder(index['model_path'], 'cuda')
        assert vision_hash == index['history_source']['vision_sha256']
        for record in group:
            state.update(phase='building', current=record['episode_id'], updated_at=time.time())
            write(status, state)
            build_cache(index, record, visual, processor)
            state['completed'] += 1
            state['samples'] += record['augmented_samples']
            write(status, state)
            print(json.dumps(dict(episode=record['episode_id'], completed=state['completed'], total=len(group))), flush=True)
        state.update(phase='completed', current=None, updated_at=time.time())
        write(status, state)
    except Exception as exc:
        state.update(phase='failed', error=repr(exc), updated_at=time.time())
        write(status, state)
        raise


def smoke(cache_dir):
    import torch
    from mem_lite.memory.vision import load_vision_encoder

    torch.set_num_threads(2)
    index = get_index(cache_dir)
    record = dict(next(row for row in index['records'] if row['dataset_kind'] == 'recovery'))
    record.update(source_frame_range=[0, 48], original_samples=9, augmented_samples=15,
                  split='train', episode_id='smoke/recovery',
                  output_dir=str(Path(cache_dir) / 'smoke/recovery'),
                  events=[dict(kind='stationary', anchor=3, frames=2, requested_frames=2),
                          dict(kind='jitter', anchor=5, frames=2, requested_frames=2)])
    visual, processor, vision_hash = load_vision_encoder(index['model_path'], 'cuda')
    assert vision_hash == index['history_source']['vision_sha256']
    build_cache(index, record, visual, processor)
    result = verify_episode(index, record)
    write(Path(cache_dir) / 'smoke-passed.json', dict(passed=True, **result))
    print(json.dumps(result), flush=True)


def validate(cache_dir):
    import torch
    from mem_lite.datasets.m3_dataset import M3Dataset

    torch.set_num_threads(1)
    cache = Path(cache_dir)
    index = get_index(cache)
    for source, expected in index['sources'].items():
        assert sha256(Path(source)) == expected, source
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda row: verify_episode(index, row), index['records']))
    for split in ('train', 'validation'):
        paths = [r['output_dir'] for r in index['records'] if r['split'] == split]
        dataset = M3Dataset(paths, load_history=False)
        assert len(dataset) == sum(row['observations'] for row in results if row['split'] == split)
        write(cache / f'{split}.sample_files.json', paths)
        for kind in ('normal', 'recovery'):
            write(cache / f'{split}.{kind}.sample_files.json', [r['output_dir'] for r in index['records']
                  if r['split'] == split and r['dataset_kind'] == kind])
    report = dict(status='completed', index_sha256=index['index_sha256'], completed_at=time.time(),
                  episode_counts=dict(Counter(r['split'] for r in results)),
                  observations={s: sum(r['observations'] for r in results if r['split'] == s)
                                for s in ('train', 'validation')},
                  history_tensors_checked=sum(r['history_tensors_checked'] for r in results),
                  images_checked=sum(r['images_checked'] for r in results),
                  history_bytes=sum(r['history_bytes'] for r in results),
                  sources_unchanged=True, histories_causal=True, labels_follow_source_frames=True,
                  three_views_synchronized=True, validation_unaugmented=True, training_started=False)
    write(cache / 'validation-report.json', report)
    write(cache / 'finetune-inputs.json', dict(checkpoint=index['checkpoint'],
          checkpoint_format='mem-lite-streamforest-v1', initialization='model weights only; fresh optimizer and cursor',
          model_path=index['model_path'], train_sample_files=str(cache/'train.sample_files.json'),
          validation_sample_files=str(cache/'validation.sample_files.json'),
          history_source=index['history_source'], history_capacity=index['history_capacity'],
          history_dim=index['history_dim'], training_started=False))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'worker', 'validate', 'smoke'])
    parser.add_argument('--plan')
    parser.add_argument('--cache-dir')
    parser.add_argument('--worker-id', type=int)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.command == 'prepare':
        if not args.plan:
            parser.error('prepare requires --plan')
        prepare(args.plan)
    else:
        if not args.cache_dir:
            parser.error('cache commands require --cache-dir')
        if args.command == 'worker':
            if args.worker_id is None or not 0 <= args.worker_id < args.workers:
                parser.error('worker requires a valid --worker-id')
            worker(args.cache_dir, args.worker_id, args.workers)
        elif args.command == 'smoke':
            smoke(args.cache_dir)
        else:
            validate(args.cache_dir)


if __name__ == '__main__':
    main()
