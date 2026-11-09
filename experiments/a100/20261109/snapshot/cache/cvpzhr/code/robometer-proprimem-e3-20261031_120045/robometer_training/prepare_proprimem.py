"""Cache Robometer pixels and causal windows from an existing Forest data split.

Only image preprocessing is cached, never Forest features or model hidden states.
The original images, PT labels and original-video split remain unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from PIL import Image
from safetensors.torch import save_file
import torch

from .model import load_processor, progress_batch


def initialize(model_path, min_pixels, max_pixels):
    global processor
    torch.set_num_threads(1)
    processor = load_processor(model_path, min_pixels=min_pixels, max_pixels=max_pixels)


def build_episode(job):
    source, output, split = map(Path, job)
    summary = json.loads((source / 'summary.json').read_text())
    episode = summary['episode_id']
    rows = [json.loads(line) for line in (source / 'samples.jsonl').read_text().splitlines()]
    target = output / 'pixels' / split / (episode + '.safetensors')
    target.parent.mkdir(parents=True, exist_ok=True)
    images = [str(source / row['images']['cam_high']) for row in rows]
    if not target.exists():
        pixel_batches, grid_batches, template = [], [], None
        for start in range(0, len(images), 16):
            frames = []
            for name in images[start:start + 16]:
                with Image.open(name) as image:
                    frames.append(image.convert('RGB'))
            result = processor.image_processor(images=frames, return_tensors='pt')
            grids = result['image_grid_thw']
            # A video has fixed geometry. The prompt depends on grids, not pixels.
            if not torch.equal(grids, grids[:1].expand_as(grids)):
                raise ValueError(f'Changing frame geometry in {episode}')
            patches = int(grids[0].prod())
            pixel_batches.append(result['pixel_values'].reshape(len(frames), patches, -1).to(torch.bfloat16))
            grid_batches.append(grids)
            if template is None:
                canonical = dict(progress_batch(processor, [(summary['task_prompt'], [frames[0]] * 8)]))
                template = {'input_' + key: value[0].contiguous()
                            for key, value in canonical.items()
                            if key not in ('pixel_values', 'image_grid_thw')}
                expected = pixel_batches[-1][0].repeat(8, 1)
                assert torch.equal(canonical['pixel_values'].to(torch.bfloat16), expected)
                assert canonical['input_ids'].shape[1] <= 4096
        grids = torch.cat(grid_batches)
        assert torch.equal(grids, grids[:1].expand_as(grids)), episode
        tensors = dict(pixels=torch.cat(pixel_batches), grids=grids, **template)
        partial = target.with_suffix('.partial')
        save_file(tensors, partial, metadata={'format': 'robometer-bf16-pixels-v1',
                  'source_samples_sha256': hashlib.sha256((source / 'samples.jsonl').read_bytes()).hexdigest()})
        partial.replace(target)
    prepared = []
    previous = -1
    for i, row in enumerate(rows):
        frame = row['frame_index']
        assert frame > previous and 0 <= row['pt'] <= 1, episode
        previous = frame
        indices = [max(0, j) for j in range(i - 7, i + 1)]
        prepared.append(dict(id=f'{episode}/{frame:08d}', episode_id=episode,
                             source_id=summary['source']['source_id'],
                             frame_index=frame, task=summary['task_prompt'], pt=row['pt'],
                             image=[images[j] for j in indices],
                             history_frame_indices=[rows[j]['frame_index'] for j in indices],
                             pixel_cache=str(target), window_indices=indices))
    return prepared, {'episode_id': episode, 'source_id': summary['source']['source_id'],
                      'task': summary['source']['task'], 'samples': len(rows),
                      'source_samples_sha256': hashlib.sha256((source / 'samples.jsonl').read_bytes()).hexdigest(),
                      'pixel_bytes': target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = {'phase': 'building', 'completed_episodes': 0, 'completed_samples': 0, 'started': time.time()}
    jobs, split_sizes = [], {}
    for split in ('train', 'validation'):
        sources = json.loads((args.source / f'{split}.sample_files.json').read_text())
        split_sizes[split] = len(sources)
        jobs.extend((source, str(args.output), split) for source in sources)
    status['total_episodes'] = len(jobs)
    def save_status():
        partial = args.output / 'status.json.tmp'
        partial.write_text(json.dumps(status, indent=2)); partial.replace(args.output / 'status.json')
    save_status()
    manifest = dict(format='robometer-proprimem-causal-pixels-v1', source=str(args.source),
                    model_path=args.model, sample_fps=5, window_size=8, min_pixels=4096,
                    max_pixels=65536, view='cam_high', padding='repeat_first_frame_on_left',
                    pixel_dtype='bfloat16', original_pt_preserved=True,
                    forest_features_used=False, splits={})
    source_sets = []
    try:
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize, initargs=(args.model, 4096, 65536)) as pool:
            results = iter(pool.map(build_episode, jobs, chunksize=1))
            for split in ('train', 'validation'):
                target = args.output / f'{split}.jsonl'
                episodes, sources = [], set()
                with target.with_suffix('.partial').open('w') as stream:
                    for _ in range(split_sizes[split]):
                        rows, info = next(results)
                        for row in rows:
                            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        episodes.append(info); sources.add(info['source_id'])
                        status['completed_episodes'] += 1
                        status['completed_samples'] += len(rows)
                        if status['completed_episodes'] % 10 == 0:
                            save_status(); print(json.dumps(status), flush=True)
                target.with_suffix('.partial').replace(target)
                manifest['splits'][split] = dict(samples=sum(e['samples'] for e in episodes),
                    episodes=episodes, original_videos=len(sources), tasks=len({e['task'] for e in episodes}),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest())
                source_sets.append(sources)
        assert not source_sets[0] & source_sets[1], 'Original video overlaps splits'
        manifest.update(status='complete', all_windows_causal=True, round_boundaries_preserved=True,
                        original_video_splits_disjoint=True, elapsed_seconds=time.time()-status['started'])
        (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        status.update(phase='completed', elapsed_seconds=manifest['elapsed_seconds']); save_status()
    except Exception as error:
        status.update(phase='failed', error=str(error)); save_status(); raise


if __name__ == '__main__':
    main()
