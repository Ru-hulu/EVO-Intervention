"""Build only selected Count caches; existing labels are immutable."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback

ROOT = Path('/cache/cvpzhr/setup/forest-count-35-legacy-e1-20261114')
sys.path.insert(0, str(ROOT / 'code'))
from experiments.prm_five_groups_20261114.prepare_cache import sha, save, progress_values, VISION_SHA
from experiments.prm_five_groups_20261114.overlap_guard import resources, pause_reasons


def main(worker_id):
    import torch
    from mem_lite.datasets.prepare_stream import build_episode, iter_observations
    from mem_lite.datasets.views import VIEW_POLICY
    from mem_lite.memory.stream import StreamMemory
    from mem_lite.memory.vision import load_vision_encoder, QwenFrameEncoder
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.35)
    status_path = ROOT / f'cache-worker-{worker_id}.json'
    state = dict(status='loading', worker=worker_id, pid=os.getpid(), gpu=worker_id % 2, completed=0, samples=0)
    save(status_path, state)
    protocol = json.loads((ROOT / 'progress-protocol.json').read_text())
    visual, processor, vision_sha = load_vision_encoder(protocol['model_path'], 'cuda')
    assert vision_sha == VISION_SHA
    memory = json.loads((ROOT / 'code/mem_lite/configs/streamforest_clip_depth.json').read_text())
    stream = StreamMemory(QwenFrameEncoder(visual), processor, memory, 65536, 4096)
    origin = stream.bind_history_source(dict(method='streamforest-v1', vision_sha256=vision_sha,
        memory_config=memory, sample_fps=5.0, include_episode_last_frame=True,
        max_pixels=65536, min_pixels=4096, view_policy=VIEW_POLICY))
    save(ROOT / f'cache-initialization-{worker_id}.json', dict(pid=os.getpid(), history_source=origin,
         vision_sha256=vision_sha, original_vision_frozen=not any(p.requires_grad for p in visual.parameters())))
    db = sqlite3.connect(ROOT / 'cache-queue.sqlite', timeout=60)
    initial_oom = resources()['oom_kill']
    while True:
        sample = resources()
        if sample['oom_kill'] > initial_oom:
            raise RuntimeError('New container OOM; preserve cache and inspect before recovery')
        reasons = pause_reasons(sample, worker_id % 2, initial_oom)
        if reasons:
            state.update(status='waiting_for_resources', reasons=reasons, updated=time.time())
            save(status_path, state)
            time.sleep(10)
            continue
        db.execute('BEGIN IMMEDIATE')
        found = db.execute('SELECT key,payload FROM jobs WHERE status="pending" ORDER BY json_extract(payload,"$.generate_missing_progress") DESC,json_extract(payload,"$.frames") DESC,key LIMIT 1').fetchone()
        if found:
            db.execute('UPDATE jobs SET status="running",worker=? WHERE key=?', (worker_id, found[0]))
        db.commit()
        if not found:
            break
        row = json.loads(found[1]); started = time.time()
        try:
            state.update(status='checking_sources', key=row['key'], updated=time.time())
            save(status_path, state)
            for path, digest in zip(row['videos'], row['video_hashes']):
                if path:
                    assert sha(path) == digest, f'Video changed: {path}'
            assert not row['generate_missing_progress'], 'Count uses verified existing visual PT only'
            progress = progress_values(row)
            output = Path(row['output'])
            assert not output.exists(), f'Preserve unexpected existing cache: {output}'
            observations = iter_observations(row['videos'], progress, sample_fps=5.0, end_frame=row['frames'])
            def reporting():
                for i, observation in enumerate(observations):
                    if i % 50 == 0:
                        state.update(status='caching', observations_in_episode=i, updated=time.time())
                        save(status_path, state)
                    yield observation
            summary = build_episode(reporting(), stream, output, row['task_prompt'], row['key'], origin, source=row)
            save(output / 'cache-verification.json', dict(source_video_hashes_verified=True,
                progress_sha256=row['progress_sha256'], samples_sha256=sha(output / 'samples.jsonl'),
                samples=summary['num_samples'], history_source=origin, finished=time.time()))
            db.execute('UPDATE jobs SET status="completed",payload=?,samples=? WHERE key=?',
                       (json.dumps(row), summary['num_samples'], row['key']))
            db.commit()
            state['completed'] += 1
            state['samples'] += summary['num_samples']
            print(json.dumps(dict(key=row['key'], samples=summary['num_samples'], seconds=time.time()-started)), flush=True)
        except Exception:
            error = traceback.format_exc()
            db.execute('UPDATE jobs SET status="failed",error=? WHERE key=?', (error, row['key']))
            db.commit()
            state.update(status='failed', error=error)
            save(status_path, state)
            raise
    state.update(status='completed', finished=time.time())
    save(status_path, state)


if __name__ == '__main__':
    main(int(sys.argv[1]))
