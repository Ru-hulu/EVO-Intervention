"""Run original-frame frozen Qwen3-VL deltas with resumable, verified outputs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'pydeps'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'model')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--memory-fraction', type=float, default=0.12)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--benchmark-frames', type=int, default=96)
    parser.add_argument('--episodes', nargs='+', help='Explicit subset; defaults to all 35')
    args = parser.parse_args()
    lock = (ROOT / 'compute.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = ROOT / ('benchmark_status.json' if args.benchmark_only else 'compute_status.json')
    status = {'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
              'phase': 'preflight', 'episodes_completed': 0, 'frames_completed': 0}
    def update(**values):
        status.update(values)
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)
    update()
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import av
        import torch
        from original_video_to_delta import QwenDelta
        if importlib.metadata.version('transformers') != '4.57.6':
            raise RuntimeError('The original Transformers 4.57.6 environment is required')
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
        source = json.loads((ROOT / 'source_manifest.json').read_text())
        if args.episodes:
            requested = set(args.episodes)
            if len(requested) != len(args.episodes) or not requested <= {x['episode'] for x in source['videos']}:
                raise ValueError('Invalid/duplicate requested episode IDs')
            source['videos'] = [x for x in source['videos'] if x['episode'] in requested]
            source['total_frames'] = sum(x['frames'] for x in source['videos'])
        update(requested_episodes=[x['episode'] for x in source['videos']], total_frames=source['total_frames'])
        model_manifest = json.loads((args.model_dir / 'download_manifest.json').read_text())
        for name, spec in model_manifest['downloaded_files'].items():
            if digest(args.model_dir / name) != spec['sha256']:
                raise RuntimeError(f'Model file changed: {name}')
        for item in source['videos']:
            video = args.dataset_root / 'videos/main_view' / (item['episode'] + '.mp4')
            if digest(video) != item['sha256']:
                raise RuntimeError(f'Video hash mismatch: {video}')
        provenance = {'model': model_manifest, 'max_pixels': 262144, 'min_pixels': 4096,
                      'dtype': 'bfloat16', 'feature': 'post-merger token FP32 mean',
                      'delta': 'adjacent original-frame FP32 L2, six decimal output, first delta zero',
                      'gpu': torch.cuda.get_device_name(0), 'memory_fraction': args.memory_fraction,
                      'original_encoder_sha256': digest(ROOT / 'original_video_to_delta.py'),
                      'versions': {key: importlib.metadata.version(key) for key in
                                   ['torch', 'torchvision', 'transformers', 'av', 'accelerate', 'safetensors']}}
        fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        provenance['fingerprint'] = fingerprint
        atomic_json(ROOT / 'provenance.json', provenance)
        output = args.dataset_root / 'delta'
        output.mkdir(parents=True, exist_ok=True)
        update(phase='loading_model')
        encoder = QwenDelta(args.model_dir, 'cuda', 262144)
        torch.cuda.synchronize()
        update(phase='benchmarking')
        benchmark_video = args.dataset_root / 'videos/main_view' / (source['videos'][0]['episode'] + '.mp4')
        start = time.perf_counter()
        previous = None
        measured = 0
        with av.open(str(benchmark_video)) as container:
            for i, frame in enumerate(container.decode(video=0)):
                current = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)
                if previous is not None:
                    value = encoder.image2delta(previous, current)
                    if not math.isfinite(value) or value < 0:
                        raise RuntimeError('Nonfinite/negative benchmark delta')
                previous = current
                measured += 1
                if measured >= args.benchmark_frames:
                    break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        benchmark = {'frames': measured, 'seconds': elapsed, 'frames_per_second': measured / elapsed,
                     'estimated_compute_minutes': source['total_frames'] / (measured / elapsed) / 60,
                     'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                     'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
        atomic_json(ROOT / 'benchmark.json', benchmark)
        print(json.dumps({'benchmark': benchmark}), flush=True)
        encoder.last_image, encoder.last_phi = None, None
        if args.benchmark_only:
            update(phase='completed', benchmark=benchmark)
            return
        update(phase='running', benchmark=benchmark)
        completed_frames = 0
        for episode_index, item in enumerate(source['videos']):
            name = item['episode']
            target = output / f'{name}-delta.txt'
            receipt_path = output / f'{name}.receipt.json'
            if target.exists() or receipt_path.exists():
                if not target.exists() or not receipt_path.exists():
                    raise RuntimeError(f'Incomplete existing output: {name}; review before resuming')
                receipt = json.loads(receipt_path.read_text())
                values = [float(line) for line in target.read_text().splitlines()]
                if (receipt['source_sha256'] != item['sha256'] or receipt['fingerprint'] != fingerprint
                    or receipt['delta_sha256'] != digest(target) or len(values) != item['frames']
                    or values[0] != 0 or not all(math.isfinite(x) and x >= 0 for x in values)):
                    raise RuntimeError(f'Existing output failed verification: {name}')
                completed_frames += len(values)
                update(episodes_completed=episode_index + 1, frames_completed=completed_frames)
                continue
            encoder.last_image, encoder.last_phi = None, None
            video = args.dataset_root / 'videos/main_view' / f'{name}.mp4'
            temporary = target.with_suffix('.txt.tmp')
            previous = None
            count = 0
            started = time.perf_counter()
            update(current_episode=name, current_episode_frames=0)
            with av.open(str(video)) as container, temporary.open('w') as stream:
                for frame in container.decode(video=0):
                    current = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)
                    delta = 0.0 if previous is None else encoder.image2delta(previous, current)
                    if not math.isfinite(delta) or delta < 0:
                        raise RuntimeError(f'Invalid delta at {name} frame {count}')
                    stream.write(f'{delta:.6f}\n')
                    previous = current
                    count += 1
                    if count % 150 == 0:
                        update(current_episode_frames=count, frames_completed=completed_frames + count)
            if count != item['frames']:
                raise RuntimeError(f'Frame count mismatch: {name}, {count} vs {item["frames"]}')
            temporary.replace(target)
            atomic_json(receipt_path, {'episode': name, 'frames': count, 'source_sha256': item['sha256'],
                                      'delta_sha256': digest(target), 'fingerprint': fingerprint,
                                      'seconds': time.perf_counter() - started})
            completed_frames += count
            update(episodes_completed=episode_index + 1, frames_completed=completed_frames)
            print(f'Completed {name}: {count} frames ({episode_index + 1}/{len(source["videos"])})', flush=True)
        update(phase='normalizing')
        from prepare_flipcup_progress import prepare_batch
        result = prepare_batch(args.dataset_root, episodes=args.episodes)
        if (result['episode_count'] != len(source['videos']) or result['total_frames'] != source['total_frames']
            or result['rollback_count'] != 2*len(source['videos'])):
            raise RuntimeError('Final manifest totals are wrong')
        update(phase='completed', result={key: value for key, value in result.items() if key != 'episodes'})
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as exc:
        update(phase='failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
