"""Rebuild the two authorized StreamForest datasets; never launch training."""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = Path('/root/autodl-tmp')
JOB = BASE / 'streamforest-jobs/rebuild-20261030'
OUT = BASE / 'streamforest-cache/rebuild-20261030'
MODEL = BASE / 'models/Qwen3.5-4B'
CODE = BASE / 'streamforest-recover-code'
OLD = BASE / 'Evo-Inter/results'
DELETE = [
    OLD / 'mem_lite/dataset_34172ee_20261026',
    OLD / 'mem_lite/progress_labels',
    OLD / 'mem_lite/streamforest_216dcd4_5fps_20261027',
    OLD / 'robobrain25/start-current-end_5fps',
    OLD / 'robometer/5fps-window8',
    OLD / 'procvlm/7tasks_5fps_w8_20261028',
    OLD / 'procvlm/labels_7tasks_20261028',
    BASE / 'streamforest-data/insert-screw-recover',
]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight():
    from mem_lite.datasets.prepare_recovery import prepare_index, probe_video
    assert not OUT.exists(), 'Output already exists; use build to resume.'
    JOB.mkdir(parents=True, exist_ok=True)
    old_split = OLD / 'mem_lite/dataset_34172ee_20261026/manifest.expert_split.json'
    split = json.loads(old_split.read_text())
    # Preserve the experimental split/provenance, not old image/feature caches.
    write(JOB / 'memlong-splits.json', {
        'source_sha256': digest(old_split), 'split': split['split'],
        'videos': [{k: row[k] for k in ['video_id', 'split', 'annotation_sha256']}
                   for row in split['videos']],
    })
    root = BASE / 'datasets/MemLong'
    metadata = {str(Path(row['video_path']).with_suffix('')): row
                for row in map(json.loads, (root / 'metadata_multiview.jsonl').read_text().splitlines())}
    records = []
    for source in split['videos']:
        if source['split'] == 'expert':
            continue
        assert source['split'] in ['train', 'validation']
        row = metadata[source['video_id']]
        annotation = root / row['annotation_path']
        assert digest(annotation) == source['annotation_sha256']
        label = json.loads(annotation.read_text())
        videos = [root / row['views'][v] for v in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']]
        probes = [probe_video(v) for v in videos]
        assert probes[0] == probes[1] == probes[2], source['video_id']
        assert probes[0]['frames'] == label['total_frames']
        assert math.isclose(probes[0]['fps'], label['fps'])
        records.append({
            'dataset': 'memlong', 'episode_id': 'memlong/' + source['video_id'],
            'source_video_id': source['video_id'], 'split': source['split'],
            'videos': list(map(str, videos)), 'video_sha256': list(map(digest, videos)),
            'annotation': str(annotation), 'annotation_sha256': digest(annotation),
            'source_frame_range': [0, label['total_frames']], 'fps': label['fps'],
            'task_prompt': label['task_prompt'],
            'progress_method': 'event_equal_progress',
            'progress_csv': str(OUT / 'labels/memlong' / source['video_id'] / (videos[0].stem + '.progress.csv')),
        })
    recovery = BASE / 'datasets/insert-screw-recover'
    # Validate and regenerate rollback labels before removing the previous derived index.
    label_dir = OUT / 'labels/insert-screw-recover'
    recovered = prepare_index(recovery, label_dir)
    sources = {r['video_id']: r for r in recovered['sources']}
    for subset in ['train', 'validation']:
        for line in (label_dir / f'{subset}.jsonl').read_text().splitlines():
            row = json.loads(line)
            assert row['source_video_id'] != 'file-019'
            row.update(dataset='insert-screw-recover', episode_id='insert-screw-recover/' + row['episode_id'],
                       video_sha256=sources[row['source_video_id']]['video_sha256'],
                       progress_method='episode_to_progress', progress_csv=str(label_dir / row['progress_csv']))
            records.append(row)
    assert len(records) == 185
    assert sum(r['split'] == 'train' for r in records) == 151
    assert sum(r['split'] == 'validation' for r in records) == 34
    ids = {s: {(r['dataset'], r['source_video_id']) for r in records if r['split'] == s}
           for s in ['train', 'validation']}
    assert ids['train'].isdisjoint(ids['validation'])
    plan = {'model_path': str(MODEL), 'sample_fps': 5, 'memory_max_pixels': 65536,
            'progress_max_pixels': 262144, 'excluded_recovery': ['file-019'],
            'excluded_memlong_experts': [r['video_id'] for r in split['videos'] if r['split'] == 'expert'],
            'records': records}
    write(JOB / 'plan.json', plan)
    print(json.dumps({'preflight': 'PASS', 'rounds': len(records), 'train': 151, 'validation': 34}), flush=True)


def cleanup():
    assert (JOB / 'plan.json').is_file()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if any(target.startswith(str(p) + '/') for p in DELETE):
                    raise RuntimeError(f'Cache in use by PID {proc.name}: {target}')
        except (FileNotFoundError, PermissionError):
            pass
    free_before = shutil.disk_usage(BASE).free
    deleted = []
    for path in DELETE:
        assert path.is_relative_to(BASE) and not path.is_symlink()
        if not path.exists():
            continue
        files = [p for p in path.rglob('*') if p.is_file()]
        deleted.append({'path': str(path), 'files': len(files), 'logical_bytes': sum(p.stat().st_size for p in files)})
        shutil.rmtree(path)
    report = {'deleted': deleted, 'freed_bytes': shutil.disk_usage(BASE).free - free_before,
              'remaining_bytes': shutil.disk_usage(BASE).free,
              'protected': ['datasets/', 'models/', 'Evo-Inter/outputs/', 'EVO-Intervention/']}
    write(JOB / 'cleanup.json', report)
    print(json.dumps(report), flush=True)


def initialize():
    global ENCODER, STREAM, ORIGIN
    import torch
    from mem_lite.memory.vision import load_vision_encoder, QwenFrameEncoder
    from mem_lite.memory.stream import StreamMemory
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.08)
    ENCODER = load_vision_encoder(MODEL, 'cuda')
    config = json.loads((CODE / 'mem_lite/configs/streamforest.json').read_text())
    STREAM = StreamMemory(QwenFrameEncoder(ENCODER[0]), ENCODER[1], config, 65536, 4096)
    ORIGIN = {'method': 'streamforest-v1', 'vision_sha256': ENCODER[2], 'memory_config': config,
              'sample_fps': 5.0, 'include_episode_last_frame': True, 'max_pixels': 65536, 'min_pixels': 4096}


def cache_one(row):
    from mem_lite.datasets.prepare_stream import build_episode, iter_observations
    from mem_lite.datasets.prepare_progress import prepare_progress
    output = OUT / row['split'] / row['episode_id']
    if (output / 'summary.json').exists():
        summary = json.loads((output / 'summary.json').read_text())
        assert summary['history_source'] == ORIGIN
        return {'episode_id': row['episode_id'], 'split': row['split'], 'path': str(output),
                'samples': summary['num_samples'], 'resumed': True}
    if shutil.disk_usage(BASE).free < 6 * 1024**3:
        raise RuntimeError('Less than 6 GiB free; stopped before starting another episode.')
    # Only discard this job's incomplete generated episode/label directory on resume.
    if output.exists():
        shutil.rmtree(output)
    csv_path = Path(row['progress_csv'])
    if row['dataset'] == 'memlong':
        metadata_path = csv_path.with_name(Path(row['videos'][0]).stem + '.metadata.json')
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            assert metadata['csv_sha256'] == digest(csv_path)
            assert metadata['annotation_sha256'] == row['annotation_sha256']
            assert metadata['vision_weights_sha256'] == ORIGIN['vision_sha256']
        else:
            if csv_path.parent.exists():
                shutil.rmtree(csv_path.parent)
            prepare_progress(row['annotation'], row['videos'][0], MODEL, csv_path.parent,
                             device='cuda', batch_size=8, max_pixels=262144, vision_encoder=ENCODER)
    else:
        assert digest(csv_path) == row['progress_sha256']
    with csv_path.open() as f:
        progress = {int(x['frame_index']): float(x['pt']) for x in csv.DictReader(f)}
    start, end = row['source_frame_range']
    assert len(progress) == end - start and math.isclose(progress[end - 1], 1)
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in progress.values())
    source = {**row, 'progress_sha256': digest(csv_path)}
    observations = iter_observations(row['videos'], progress, sample_fps=5, start_frame=start, end_frame=end)
    summary = build_episode(observations, STREAM, output, row['task_prompt'], row['episode_id'], ORIGIN, source=source)
    return {'episode_id': row['episode_id'], 'split': row['split'], 'path': str(output), 'samples': summary['num_samples']}


def verify(records):
    import torch
    from mem_lite.datasets.m3_dataset import M3Dataset
    outputs = {s: [str(OUT / s / r['episode_id']) for r in records if r['split'] == s]
               for s in ['train', 'validation']}
    checks = {'episodes': 0, 'images': 0, 'histories': 0, 'loaded_samples': 0, 'splits': {}}
    for split, paths in outputs.items():
        dataset = M3Dataset(paths)
        checks['splits'][split] = {'episodes': len(paths), 'samples': len(dataset)}
        offset = 0
        for directory, summary in dataset.episodes:
            rows = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
            assert len(rows) == summary['num_samples'] and rows[0]['history'] is None
            assert rows[-1]['frame_index'] == summary['source']['source_frame_range'][1] - 1
            assert rows[-1]['pt'] == 1
            for i, row in enumerate(rows):
                for image in row['images'].values():
                    assert (directory / image).is_file()
                    checks['images'] += 1
                if i:
                    assert row['history'] and (directory / row['history']).stat().st_size > 0
                    checks['histories'] += 1
            for local in sorted({0, 1, len(rows)//2, len(rows)-1}):
                sample = dataset[offset + local]
                tokens = sample['history_tokens']
                if local:
                    assert tokens.ndim == 2 and torch.isfinite(tokens).all()
                    assert tokens.shape[0] <= dataset.history_capacity
                else:
                    assert tokens is None
                checks['loaded_samples'] += 1
            offset += len(rows)
            checks['episodes'] += 1
        write(OUT / f'{split}.sample_files.json', paths)
    config = json.loads((CODE / 'mem_lite/configs/stage3_train.json').read_text())
    config.update(model_path=str(MODEL), sample_files=outputs['train'],
                  output_dir=str(BASE / 'Evo-Inter/outputs/mem_lite/streamforest_recovery_memlong_20261030'))
    write(OUT / 'train.config.json', config)
    checks.update(status='PASS', training_started=False, free_bytes=shutil.disk_usage(BASE).free)
    write(OUT / 'verification.json', checks)
    write(OUT / 'manifest.json', {'plan': str(JOB / 'plan.json'), 'source_datasets': ['insert-screw-recover', 'MemLong'],
                                 'samples': outputs, 'summary': checks, 'training_started': False})
    return checks


def build(workers):
    plan = json.loads((JOB / 'plan.json').read_text())
    # Alternate datasets so both exercise the pipeline early; long episodes first within each group.
    records = sorted(plan['records'], key=lambda r: (r['dataset'], -(r['source_frame_range'][1]-r['source_frame_range'][0])))
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'total': len(records), 'completed': 0,
             'workers': workers, 'samples': 0, 'results': [], 'training_started': False}
    started = time.time()
    write(JOB / 'status.json', state)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
            futures = {pool.submit(cache_one, row): row for row in records}
            for future in as_completed(futures):
                result = future.result()
                state['results'].append(result)
                state.update(completed=len(state['results']), samples=sum(r['samples'] for r in state['results']),
                             elapsed_seconds=time.time()-started, free_bytes=shutil.disk_usage(BASE).free)
                write(JOB / 'status.json', state)
                print(json.dumps({k: state[k] for k in ['completed','total','samples','elapsed_seconds','free_bytes']}), flush=True)
        state['status'] = 'VERIFYING'
        write(JOB / 'status.json', state)
        state['verification'] = verify(plan['records'])
        state['status'] = 'COMPLETE'
    except BaseException as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['elapsed_seconds'] = time.time()-started
        write(JOB / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preflight','cleanup','build'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.workers)
    else:
        globals()[args.command]()
