"""Build the fixed screw split with duration-based holds and closed reverse/forward loops."""

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

from mem_lite.datasets.prepare_recovery import CAMERA_DIRS, probe_video, sha256, write_json
from scripts.delta_to_progress import episode_to_progress

FORMAT = "streamforest-screw-mixed-augmentation-v1"


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def combine_rounds(rounds):
    """Keep original labels, including rollback; retain inter-round idle frames."""
    values, start = [], rounds[0][0]
    for k, (left, right, progress) in enumerate(rounds):
        if left < start + len(values) or len(progress) != right - left:
            raise ValueError("Overlapping rounds or incorrect label length")
        if not progress or not math.isclose(progress[-1], 1.0):
            raise ValueError("Each original round must finish at PT=1")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in progress):
            raise ValueError("Invalid original PT")
        values.extend([k / len(rounds)] * (left - start - len(values)))
        values.extend((k + p) / len(rounds) for p in progress)
    return values


def sample_indices(frame_count, fps):
    step = Fraction(str(fps)) / 5
    indices, n = [], 0
    while math.ceil(n * step) < frame_count:
        indices.append(math.ceil(n * step))
        n += 1
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def choose_events(indices, frame_count, fps, seed, episode_id, enabled=True):
    if not enabled:
        return []
    count = math.ceil(Fraction(frame_count, 1) / Fraction(str(fps)) / 15)
    rng = random.Random(int(digest([seed, episode_id]), 16))
    events = []
    for kind, maximum in (("stationary", 25), ("jitter", 10)):
        for anchor in sorted(rng.sample(range(len(indices)), count)):
            requested = rng.randint(0, maximum)
            frames = min(requested, anchor) if kind == "jitter" else requested
            events.append(dict(kind=kind, anchor=anchor, frames=frames, requested_frames=requested))
    return events


def expand_schedule(indices, fps, events):
    by_anchor = defaultdict(list)
    for event in events:
        by_anchor[event['anchor']].append(event)
    step, added = Fraction(str(fps)) / 5, 0
    for i, original in enumerate(indices):
        position = Fraction(original) + added * step
        yield dict(source_sample_index=i, frame_index=float(position), kind="original")
        for event in sorted(by_anchor[i], key=lambda e: e['kind'] != 'stationary'):
            n = event['frames']
            if event['kind'] == 'stationary':
                path = [(i, 'stationary')] * n
            else:
                path = ([(j, 'reverse') for j in range(i - 1, i - n - 1, -1)]
                        + [(j, 'forward') for j in range(i - n + 1, i + 1)])
            for source, kind in path:
                added += 1
                position += step
                yield dict(source_sample_index=source, frame_index=float(position), kind=kind)


def make_index(plan_path, cache_dir, reference_config, model_path):
    cache = Path(cache_dir)
    if (cache / 'index.json').exists():
        return get_index(cache)
    plan, reference = read(plan_path), read(reference_config)
    for kind in ('stationary', 'jitter'):
        assert plan['augmentation'][kind]['anchor_count_formula'] == 'ceil(unaugmented_episode_duration_seconds / 15)'
    seen = set(plan['base_checkpoint_seen_normal_videos'])
    validation = set(plan['normal_validation_videos'])
    assert not seen & validation
    normal = Path(plan['normal_source'])
    records, inputs, video_info, normal_rounds = [], {}, {}, {}

    def protect(path):
        path = Path(path)
        if str(path) not in inputs:
            inputs[str(path)] = sha256(path)

    def videos_at(paths):
        for path in paths:
            if path not in video_info:
                video_info[path] = probe_video(path)
                protect(path)
        metas = [video_info[p] for p in paths]
        if any(m != metas[0] for m in metas):
            raise ValueError(f"Unsynchronized three-view source: {paths}")
        return metas[0]

    def add(task, video, episodes, split, paths, start, end, values, boundaries):
        meta = videos_at(paths)
        assert 0 <= start < end <= meta['frames'] and len(values) == end - start
        episode_id = f"{task}/{video}/n{len(episodes)}-" + '-'.join(map(str, episodes))
        indices = sample_indices(end - start, meta['fps'])
        events = choose_events(indices, end - start, meta['fps'], plan['random_seed'], episode_id, split == 'train')
        added = sum(e['frames'] * (2 if e['kind'] == 'jitter' else 1) for e in events)
        csv_path = cache / 'labels' / episode_id / 'progress.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_index', 'source_frame_index', 'pt'])
            writer.writerows((i, start+i, p) for i, p in enumerate(values))
        records.append(dict(episode_id=episode_id, task=task, source_video_id=video,
                            source_episodes=episodes, split=split, videos=paths,
                            source_frame_range=[start, end], fps=meta['fps'],
                            round_boundaries=boundaries, task_prompt=plan['task_prompts'][str(len(episodes))],
                            progress_csv=str(csv_path), progress_sha256=sha256(csv_path),
                            original_samples=len(indices), augmented_samples=len(indices)+added,
                            duration_seconds=(end-start)/meta['fps'], events=events,
                            output_dir=str(cache / split / episode_id)))

    for split, key in [('train', 'normal_train_episodes'), ('validation', 'normal_validation_episodes')]:
        for row in plan[key]:
            video = row['source_video_id']
            assert (video in validation) == (split == 'validation')
            if video not in normal_rounds:
                label = next(p for p in (normal/'splitlable').iterdir()
                             if p.stem == video and p.suffix.lower() == '.json')
                delta = normal/'delta'/f'{video}-delta.txt'
                protect(label); protect(delta)
                annotation = read(label)
                deltas = [float(line.split()[0]) for line in delta.read_text().splitlines() if line.strip()]
                paths = [str(normal/c/f'{video}.mp4') for c in CAMERA_DIRS]
                assert len(deltas) == videos_at(paths)['frames']
                task_ids = [t['id'] for t in annotation['taskid']]
                rounds = {}
                for task in annotation['tasks']:
                    for e in task['episodes']:
                        ss = e['subtasks']
                        pt, _ = episode_to_progress(deltas, ss, task_ids)
                        rounds[e['episode']] = (ss[0]['30FPSstart_frame'], ss[-1]['30FPSend_frame']+1, pt)
                normal_rounds[video] = rounds
            selected = [normal_rounds[video][i] for i in row['source_episodes']]
            start, end = selected[0][0], selected[-1][1]
            add('insert-screw', video, row['source_episodes'], split,
                [str(normal/c/f'{video}.mp4') for c in CAMERA_DIRS], start, end,
                combine_rounds(selected), [[a-start, b-start] for a,b,_ in selected])
    prepared = Path(plan['recovery']['prepared_index'])
    for split in ('train', 'validation'):
        source = prepared / f'{split}.jsonl'
        protect(source)
        for line in source.read_text().splitlines():
            row = json.loads(line)
            label = prepared / row['progress_csv']
            protect(label)
            assert sha256(label) == row['progress_sha256']
            with label.open() as f:
                values = [float(r['pt']) for r in csv.DictReader(f)]
            start, end = row['source_frame_range']
            add('insert-screw-recover', row['source_video_id'], row['source_episodes'], split,
                row['source_videos'], start, end, values,
                [[0, row['first_round_end']], [row['second_round_start'], end-start]])
    counts = Counter(r['split'] for r in records)
    assert counts == {'train':277, 'validation':136}
    train_sources = {(r['task'], r['source_video_id']) for r in records if r['split']=='train'}
    val_sources = {(r['task'], r['source_video_id']) for r in records if r['split']=='validation'}
    assert not train_sources & val_sources
    # Also exclude any validation source used by the initialization checkpoint.
    old_files = reference['sample_files']
    if isinstance(old_files, str):
        old_files = read(old_files)
    for p in old_files:
        directory = Path(p) if Path(p).is_dir() else Path(p).parent
        src = read(directory/'summary.json').get('source', {})
        assert (src.get('task'), src.get('video_id')) not in val_sources
    index = dict(format=FORMAT, plan_path=str(plan_path), plan_sha256=sha256(Path(plan_path)),
                 reference_config=str(reference_config), model_path=str(model_path),
                 history_source=reference['history_source'], history_dim=reference['model']['history_dim'],
                 history_capacity=reference['history_capacity'], sources=inputs, records=records)
    index['index_sha256'] = digest(index)
    write_json(cache/'index.json', index)
    write_json(cache/'index-summary.json', dict(episode_counts=dict(counts),
               original_observations={s:sum(r['original_samples'] for r in records if r['split']==s) for s in counts},
               observations={s:sum(r['augmented_samples'] for r in records if r['split']==s) for s in counts},
               anchors=dict(Counter(e['kind'] for r in records for e in r['events'])), training_started=False))
    return index


def get_index(cache):
    index = read(Path(cache)/'index.json')
    assert index['format'] == FORMAT
    assert index['index_sha256'] == digest({k:v for k,v in index.items() if k!='index_sha256'})
    assert sha256(Path(index['plan_path'])) == index['plan_sha256']
    return index


def raw_features(visual, processor, image, settings):
    import torch
    inputs = processor(images=[image], return_tensors='pt', max_pixels=settings['max_pixels'],
                       min_pixels=settings['min_pixels'])
    parameter = next(visual.parameters())
    grid = inputs['image_grid_thw'].to(parameter.device)
    pixels = inputs['pixel_values'].to(device=parameter.device, dtype=parameter.dtype)
    with torch.no_grad(), torch.autocast(parameter.device.type, enabled=False):
        features = visual(pixels, grid_thw=grid, return_dict=True).pooler_output
    assert len(grid)==1 and int(grid[0,0])==1 and torch.isfinite(features).all().item()
    merge = visual.config.spatial_merge_size
    return features, int(grid[0,1])//merge, int(grid[0,2])//merge


def build_cache(index, record, visual, processor):
    import torch
    from mem_lite.datasets.m3_dataset import DATA_FORMAT, VIEW_ORDER
    from mem_lite.datasets.prepare_stream import iter_observations
    from mem_lite.memory.manager import MemoryManager
    from mem_lite.memory.vision import spatiotemporal_positions

    output = Path(record['output_dir'])
    if (output/'summary.json').exists():
        assert read(output/'summary.json')['source'] == record
        return
    partial = output.with_name(output.name+'.partial')
    ownership = dict(index_sha256=index['index_sha256'], episode_id=record['episode_id'])
    if partial.exists():
        assert read(partial/'owner.json') == ownership
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    write_json(partial/'owner.json', ownership)
    for name in (*VIEW_ORDER, 'history'):
        (partial/name).mkdir()
    assert sha256(Path(record['progress_csv'])) == record['progress_sha256']
    with Path(record['progress_csv']).open() as f:
        labels = [float(r['pt']) for r in csv.DictReader(f)]
    start, end = record['source_frame_range']
    progress = {start+i:p for i,p in enumerate(labels)}
    indices = sample_indices(end-start, record['fps'])
    observations = iter_observations(record['videos'], progress, sample_fps=5.0, start_frame=start, end_frame=end)
    memory = MemoryManager(index['history_dim'], **index['history_source']['memory_config'])
    assert memory.token_budget == index['history_capacity']
    recent, last_original, previous = {}, -1, None
    started, max_tokens = time.monotonic(), 0
    with (partial/'samples.jsonl').open('w') as f:
        for count, step in enumerate(expand_schedule(indices, record['fps'], record['events']), 1):
            source = step['source_sample_index']
            if step['kind']=='original':
                assert source == last_original+1
                observation = next(observations)
                assert observation['frame_index'] == start+indices[source]
                images = {}
                for view, image in zip(VIEW_ORDER, observation['images'], strict=True):
                    assert image is not None
                    relative = f'{view}/{observation["frame_index"]:08d}.png'
                    image.save(partial/relative, compress_level=1)
                    images[view] = relative
                features = raw_features(visual, processor, observation['images'][0], index['history_source'])
                recent[source] = (images, features)
                recent.pop(source-11, None)
                last_original = source
            images, (features, height, width) = recent[source]
            # The snapshot excludes the current frame. PT never enters the memory API.
            history = memory.get_memory_tokens()
            tokens = 0 if history is None else len(history)
            assert tokens <= index['history_capacity']
            history_path = None
            if history is not None:
                history_path = f'history/{count-1:08d}.pt'
                torch.save(dict(tokens=history.cpu(), through_frame=previous), partial/history_path)
            row = dict(**step, sample_index=count-1, source_frame_index=start+indices[source],
                       timestamp_seconds=step['frame_index']/record['fps'], images=images,
                       pt=labels[indices[source]], history=history_path, history_token_count=tokens)
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
            positions = spatiotemporal_positions(height, width, step['frame_index'], index['history_dim'], features.device)
            memory.update(features + positions.to(features.dtype))
            previous, max_tokens = step['frame_index'], max(max_tokens, tokens)
    assert count == record['augmented_samples'] and last_original+1 == record['original_samples']
    assert next(observations, None) is None
    summary = dict(format_version=DATA_FORMAT, episode_id=record['episode_id'], task_prompt=record['task_prompt'],
                   num_samples=count, history_dim=index['history_dim'], history_capacity=index['history_capacity'],
                   history_source=index['history_source'], source=record,
                   augmentation=dict(index_sha256=index['index_sha256'], events=record['events'],
                                     frame_index_unit='source-fps local frame-equivalent position'),
                   generation=dict(seconds=time.monotonic()-started, raw_visual_forward_calls=last_original+1,
                                   memory_updates=count, max_history_tokens=max_tokens))
    write_json(partial/'summary.json', summary)
    partial.rename(output)


def assigned(index, workers):
    groups, sizes = [[] for _ in range(workers)], [0]*workers
    for row in sorted(index['records'], key=lambda r:(-r['augmented_samples'], r['episode_id'])):
        i = min(range(workers), key=lambda j:sizes[j])
        groups[i].append(row)
        sizes[i] += row['augmented_samples']
    return groups


def worker(cache_dir, worker_id, workers, limit=None):
    import fcntl
    import torch
    from mem_lite.memory.vision import load_vision_encoder
    torch.set_num_threads(2)
    cache = Path(cache_dir)
    lock = (cache/f'worker-{worker_id}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    index = get_index(cache)
    group = assigned(index, workers)[worker_id]
    if limit is not None:
        group = group[:limit]
    state = dict(pid=os.getpid(), phase='loading_vision', completed=0, samples=0, total=len(group))
    status = cache/f'worker-{worker_id}.json'
    write_json(status, state)
    try:
        visual, processor, vision_hash = load_vision_encoder(index['model_path'], 'cuda')
        assert vision_hash == index['history_source']['vision_sha256']
        for row in group:
            state.update(phase='building', current=row['episode_id'], updated_at=time.time())
            write_json(status, state)
            build_cache(index, row, visual, processor)
            state['completed'] += 1
            state['samples'] += row['augmented_samples']
            print(json.dumps(dict(episode=row['episode_id'], samples=row['augmented_samples'], completed=state['completed'], total=len(group))), flush=True)
        state.update(phase='completed', current=None, updated_at=time.time())
        write_json(status, state)
    except Exception as exc:
        state.update(phase='failed', error=repr(exc), updated_at=time.time())
        write_json(status, state)
        raise


def verify_episode(index, record):
    import torch
    from mem_lite.datasets.m3_dataset import VIEW_ORDER
    directory = Path(record['output_dir'])
    summary = read(directory/'summary.json')
    assert summary['source']==record and summary['history_source']==index['history_source']
    assert summary['augmentation']['index_sha256']==index['index_sha256']
    with Path(record['progress_csv']).open() as f:
        labels = [float(r['pt']) for r in csv.DictReader(f)]
    assert sha256(Path(record['progress_csv']))==record['progress_sha256']
    start, end = record['source_frame_range']
    indices = sample_indices(end-start, record['fps'])
    rows = [json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
    assert len(rows)==summary['num_samples']==record['augmented_samples']
    assert [r['source_sample_index'] for r in rows if r['kind']=='original']==list(range(len(indices)))
    expected = list(expand_schedule(indices, record['fps'], record['events']))
    previous, checked_images, history_bytes = None, set(), 0
    for i,(row, step) in enumerate(zip(rows, expected, strict=True)):
        assert all(row[k]==v for k,v in step.items()) and row['sample_index']==i
        source = indices[row['source_sample_index']]
        assert row['source_frame_index']==start+source and row['pt']==labels[source]
        assert math.isfinite(row['pt']) and 0<=row['pt']<=1
        assert previous is None or row['frame_index']>previous
        assert math.isclose(row['timestamp_seconds'], row['frame_index']/record['fps'])
        if row['kind']!='original':
            assert math.isclose((row['frame_index']-previous)/record['fps'], 0.2)
        for view in VIEW_ORDER:
            assert row['images'][view]==f'{view}/{start+source:08d}.png'
            path = directory/row['images'][view]
            if path not in checked_images:
                assert path.is_file() and path.stat().st_size>0
                checked_images.add(path)
        assert (row['history'] is None)==(i==0)
        if row['history'] is not None:
            path = directory/row['history']
            h = torch.load(path, map_location='cpu', weights_only=True)
            t = h['tokens']
            assert h['through_frame']==previous
            assert t.ndim==2 and t.shape[1]==index['history_dim'] and 0<len(t)<=index['history_capacity']
            assert len(t)==row['history_token_count'] and t.dtype==torch.bfloat16 and torch.isfinite(t).all().item()
            history_bytes += path.stat().st_size
        previous = row['frame_index']
    assert len(list((directory/'history').glob('*.pt')))==len(rows)-1
    assert len(checked_images)==3*record['original_samples']
    counts = Counter(e['kind'] for e in record['events'])
    if record['split']=='train':
        n = math.ceil(Fraction(end-start,1)/Fraction(str(record['fps']))/15)
        assert counts=={'stationary':n,'jitter':n}
    else:
        assert not record['events'] and len(rows)==len(indices)
    return dict(episode_id=record['episode_id'], split=record['split'], task=record['task'],
                rounds=len(record['source_episodes']), observations=len(rows),
                history_tensors_checked=len(rows)-1, images_checked=len(checked_images), history_bytes=history_bytes)


def validate(cache_dir, workers=8):
    import torch
    from mem_lite.datasets.m3_dataset import M3Dataset
    torch.set_num_threads(1)
    cache = Path(cache_dir)
    index = get_index(cache)
    for path, expected in index['sources'].items():
        assert sha256(Path(path))==expected, path
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results=[]
        for result in pool.map(lambda r:verify_episode(index,r), index['records']):
            results.append(result)
            if len(results)%20==0:
                print(json.dumps(dict(verified=len(results), total=len(index['records']))), flush=True)
    for split in ('train','validation'):
        paths=[r['output_dir'] for r in index['records'] if r['split']==split]
        dataset=M3Dataset(paths, load_history=False)
        assert len(dataset)==sum(r['observations'] for r in results if r['split']==split)
        write_json(cache/f'{split}.sample_files.json', paths)
        for task in ('insert-screw','insert-screw-recover'):
            for n in (1,2,3):
                subset=[r['output_dir'] for r in index['records'] if r['split']==split and r['task']==task and len(r['source_episodes'])==n]
                if subset:
                    write_json(cache/f'{split}.{task}.n{n}.sample_files.json', subset)
    report=dict(status='completed', index_sha256=index['index_sha256'], completed_at=time.time(),
                episode_counts=dict(Counter(r['split'] for r in results)),
                observations={s:sum(r['observations'] for r in results if r['split']==s) for s in ('train','validation')},
                history_tensors_checked=sum(r['history_tensors_checked'] for r in results),
                images_checked=sum(r['images_checked'] for r in results),
                history_bytes=sum(r['history_bytes'] for r in results),
                sources_unchanged=True, histories_causal=True, labels_follow_source_frames=True,
                three_views_synchronized=True, validation_unaugmented=True, training_started=False)
    write_json(cache/'validation-report.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['index','worker','validate'])
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--reference-config')
    parser.add_argument('--model-path')
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args=parser.parse_args()
    if args.command=='index':
        make_index(args.plan, args.cache_dir, args.reference_config, args.model_path)
        print(json.dumps(read(Path(args.cache_dir)/'index-summary.json')), flush=True)
    elif args.command=='worker':
        worker(args.cache_dir, args.worker_id, args.workers, args.limit)
    else:
        validate(args.cache_dir, args.workers)


if __name__=='__main__':
    main()
