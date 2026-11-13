"""Frozen-manifest FoREST cache preparation; no labels or splits are modified."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import traceback

BASE = Path('/cache/cvpzhr/setup/prm-five-groups-20261114')
DATA = Path('/cache/cvpzhr/datasets/ProPriMem')
CACHE = Path('/cache/cvpzhr/datasets/forest-four-suites-clip-depth-5fps-20261114')
MODEL = '/cache/cvpzhr/models/Qwen3.5-4B'
VISION_SHA = '1cc7c705e96c07fb52bcb1fb32f5e05271cc249fbc619c1128fd4eddbea47aea'
RECOVER_PROMPTS = {
    'R01': 'Flip the two cups one at a time from left to right. Turn each cup over and then flip it back to its original orientation before moving on to the next cup.',
    'R02': 'Inspect each of the three pencil cases. Pick it up, open and close its zipper, then put it back.',
    'R03': 'Pick up each of the three spray bottles from the rack, set it on the table, then return it to its original position in the rack.',
    'R04': 'Open each of the three drawers in any order, look inside, and close it again.',
    'R05': 'Pour water into each of three identical cups in any order.',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def connect():
    db = sqlite3.connect(BASE / 'cache-queue.sqlite', timeout=60)
    db.row_factory = sqlite3.Row
    return db


def progress_values(row):
    path = Path(row['progress_path'])
    if sha(path) != row['progress_sha256']:
        raise ValueError(f'Progress changed: {path}')
    if path.suffix == '.csv':
        with path.open() as f:
            values = list(csv.DictReader(f))
    else:
        label = json.loads(path.read_text())
        if label['video_sha256'] != row['video_hashes'][0]:
            raise ValueError(f'Progress video hash mismatch: {path}')
        values = label['frames']
    if [int(x['frame_index']) for x in values] != list(range(row['frames'])):
        raise ValueError(f'Incomplete original frame labels: {path}')
    values = [float(x['pt']) for x in values]
    if not all(0 <= x <= 1 for x in values):
        raise ValueError(f'Invalid progress: {path}')
    return dict(enumerate(values))


def make_index(append_recover=False):
    BASE.mkdir(parents=True, exist_ok=True)
    if (BASE / 'cache-queue.sqlite').exists() and not append_recover:
        raise FileExistsError('Existing queue: inspect it instead of initializing twice')
    plan = json.loads((Path(__file__).parent / 'experiment-plan.json').read_text())
    records, blocked = [], []
    for suite in (['Recover'] if append_recover else ['State', 'Count', 'Order', 'Recover']):
        root = DATA / suite
        manifest = root / 'manifest.json'
        if sha(manifest) != plan['datasets']['suites'][suite]['manifest_sha256']:
            raise ValueError(f'Frozen manifest changed: {suite}')
        for task in json.loads(manifest.read_text())['tasks']:
            if suite == 'Recover' and task['task_id'] not in plan['datasets']['recover_task_ids']:
                continue
            meta = root / task['trajectory_index']
            if sha(meta) != task['trajectory_index_sha256']:
                raise ValueError(f'Metadata hash mismatch: {meta}')
            rows = {r['sample_id']: r for r in map(json.loads, meta.read_text().splitlines())}
            for split in ['train', 'val']:
                for sample in task[split]:
                    r = rows[sample]
                    episode = f"episode_{int(r['episode_index']):06d}"
                    progress = meta.parent / (r.get('progress_path') or f'annotation/{episode}.progress.json')
                    if suite == 'Recover' and task['task_id'] == 'R05' and not progress.exists():
                        progress = BASE / 'labels/Recover/R05' / f'{episode}.progress.csv'
                    if not progress.exists():
                        blocked.append({'suite': suite, 'task_id': task['task_id'], 'sample_id': sample,
                                        'split': split, 'reason': 'missing_progress', 'expected': str(progress)})
                        continue
                    # Recover needs goal-only instructions, not a narration of the scripted failure.
                    prompt = RECOVER_PROMPTS[task['task_id']] if suite == 'Recover' else r.get('task_prompt')
                    prompt = prompt or r.get('source_instruction')
                    if not prompt:
                        raise ValueError(f'Missing task instruction: {sample}')
                    views = r['video_views']
                    order = ['main_view', 'wrist_view', None] if 'wrist_view' in views else ['main_view', 'left_view', 'right_view']
                    videos = [str(meta.parent / views[v]['video_path']) if v in views else None for v in order]
                    hashes = [views[v]['video_sha256'] if v in views else None for v in order]
                    key = f"{suite}/{task['task_id']}/{sample}"
                    row = {'key': key, 'suite': suite, 'task_id': task['task_id'], 'sample_id': sample,
                           'split': split, 'frames': r['total_frames'], 'fps': r['fps'],
                           'videos': videos, 'video_hashes': hashes, 'progress_path': str(progress),
                           'progress_sha256': sha(progress), 'task_prompt': prompt,
                           'output': str(CACHE / split / key), 'metadata_sha256': sha(meta)}
                    if r.get('progress_sha256') and row['progress_sha256'] != r['progress_sha256']:
                        raise ValueError(f'Label receipt mismatch: {sample}')
                    progress_values(row)
                    records.append(row)
    db = connect()
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, suite TEXT, split TEXT, priority INTEGER, frames INTEGER, payload TEXT, status TEXT DEFAULT "pending", worker TEXT, started REAL, finished REAL, samples INTEGER, error TEXT)')
    priorities = {'State': 0, 'Count': 1, 'Order': 2, 'Recover': 3}
    with db:
        for r in records:
            old = db.execute('SELECT payload FROM jobs WHERE key=?', (r['key'],)).fetchone()
            if old and json.loads(old['payload']) != r:
                raise ValueError(f'Cannot change an existing frozen cache input: {r["key"]}')
        db.executemany('INSERT OR IGNORE INTO jobs(key,suite,split,priority,frames,payload) VALUES(?,?,?,?,?,?)',
                       [(r['key'], r['suite'], r['split'], priorities[r['suite']] + (10 if r['split']=='val' else 0), r['frames'], json.dumps(r)) for r in records])
    save(BASE / ('recover-input-index.json' if append_recover else 'cache-input-index.json'), {'records': records, 'blocked': blocked,
         'plan_sha256': sha(Path(__file__).parent / 'experiment-plan.json'),
         'source_labels_unchanged': True, 'splits_unchanged': True})
    save(BASE / 'data-blockers.json', {'blocked': blocked})
    print(json.dumps({'indexed': len(records), 'blocked': len(blocked)}), flush=True)


def claim(db, worker):
    db.execute('BEGIN IMMEDIATE')
    row = db.execute('SELECT key,payload FROM jobs WHERE status="pending" ORDER BY priority,frames DESC,key LIMIT 1').fetchone()
    if row:
        db.execute('UPDATE jobs SET status="running", worker=?, started=? WHERE key=?', (worker, time.time(), row['key']))
    db.commit()
    return json.loads(row['payload']) if row else None


def worker(worker_id, gpu):
    import torch
    from mem_lite.datasets.prepare_stream import build_episode, iter_observations
    from mem_lite.datasets.views import VIEW_POLICY
    from mem_lite.memory.stream import StreamMemory
    from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder
    torch.set_num_threads(2)
    status = BASE / f'cache-worker-{worker_id}.json'
    save(status, {'phase': 'loading', 'pid': os.getpid(), 'gpu': gpu, 'started': time.time()})
    visual, processor, vision_hash = load_vision_encoder(MODEL, 'cuda')
    assert vision_hash == VISION_SHA
    memory = json.loads((Path(__file__).parents[2] / 'mem_lite/configs/streamforest_clip_depth.json').read_text())
    stream = StreamMemory(QwenFrameEncoder(visual), processor, memory, 65536, 4096)
    origin = stream.bind_history_source({'method': 'streamforest-v1', 'vision_sha256': vision_hash,
        'memory_config': memory, 'sample_fps': 5.0, 'include_episode_last_frame': True,
        'max_pixels': 65536, 'min_pixels': 4096, 'view_policy': VIEW_POLICY})
    save(BASE / f'cache-initialization-{worker_id}.json', {'pid': os.getpid(), 'gpu': gpu, 'history_source': origin})
    db, completed = connect(), 0
    while not (BASE / f'drain-gpu-{gpu}').exists():
        row = claim(db, worker_id)
        if row is None:
            break
        start = time.time()
        try:
            progress = progress_values(row)
            for path, digest in zip(row['videos'], row['video_hashes']):
                if path and sha(path) != digest:
                    raise ValueError(f'Video changed: {path}')
            output = Path(row['output'])
            if output.exists():
                raise FileExistsError(f'Unreviewed existing output; preserve for recovery: {output}')
            observations = iter_observations(row['videos'], progress, sample_fps=5.0, end_frame=row['frames'])
            def reporting():
                for i, obs in enumerate(observations):
                    if i % 50 == 0:
                        save(status, {'phase': 'cache', 'pid': os.getpid(), 'gpu': gpu, 'key': row['key'],
                                      'frame': obs['frame_index'], 'samples_in_episode': i,
                                      'completed': completed, 'updated': time.time()})
                    yield obs
            result = build_episode(reporting(), stream, output, row['task_prompt'], row['key'], origin, source=row)
            save(output / 'cache-verification.json', {'source_video_hashes_verified': True,
                 'progress_sha256': row['progress_sha256'], 'samples_sha256': sha(output / 'samples.jsonl'),
                 'samples': result['num_samples'], 'history_source': origin, 'finished': time.time()})
            with db:
                db.execute('UPDATE jobs SET status="completed", finished=?, samples=? WHERE key=?',
                           (time.time(), result['num_samples'], row['key']))
            completed += 1
            print(json.dumps({'key': row['key'], 'samples': result['num_samples'], 'seconds': time.time()-start}), flush=True)
        except Exception:
            error = traceback.format_exc()
            with db:
                db.execute('UPDATE jobs SET status="failed",error=?,finished=? WHERE key=?', (error,time.time(),row['key']))
            save(status, {'phase': 'failed', 'pid': os.getpid(), 'gpu': gpu, 'key': row['key'], 'error': error})
            raise
    save(status, {'phase': 'exited', 'pid': os.getpid(), 'gpu': gpu, 'completed': completed, 'finished': time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['index','append-recover','worker'])
    parser.add_argument('--worker-id')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.mode in ['index', 'append-recover']:
        make_index(args.mode == 'append-recover')
    else:
        worker(args.worker_id, args.gpu)


if __name__ == '__main__':
    main()
