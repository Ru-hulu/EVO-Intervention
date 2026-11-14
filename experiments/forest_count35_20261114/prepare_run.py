"""Freeze 35 Count training trajectories, with at least two per task."""
import collections
import json
from pathlib import Path
import random
import shutil
import sqlite3
import time

from experiments.prm_five_groups_20261114.prepare_cache import sha, save, progress_values

ROOT = Path('/cache/cvpzhr/setup/forest-count-35-legacy-e1-20261114')
DATA = Path('/cache/cvpzhr/datasets/ProPriMem/Count')
CACHE = Path('/cache/cvpzhr/datasets/forest-count-35-clip-depth-5fps-20261114/train')
REFERENCE = Path('/cache/cvpzhr/setup/forest-order-35-legacy-e1-20261114')
OUTPUT = Path('/cache/cvpzhr/outputs/forest-count-35-legacy-2gpu-e1-20261114')
LABEL_SETUP = Path('/cache/cvpzhr/setup/proprimem-visual-pt-20261113')


def select(groups, total=35, seed=42):
    """Two per task, then one extra in each randomly chosen eligible task."""
    if not groups or any(len(v) != len(set(v)) or len(v) < 2 for v in groups.values()):
        raise ValueError('Each task needs at least two distinct training episodes')
    extras = total - 2 * len(groups)
    eligible = sorted(k for k, v in groups.items() if len(v) > 2)
    if not 0 <= extras <= len(eligible):
        raise ValueError('Requested total cannot use two or three episodes per task')
    rng = random.Random(seed)
    selected = {k: rng.sample(sorted(groups[k]), 2) for k in sorted(groups)}
    for task in rng.sample(eligible, extras):
        selected[task].append(rng.choice(sorted(set(groups[task]) - set(selected[task]))))
    return {k: sorted(v) for k, v in selected.items()}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'cache-queue.sqlite').exists(), 'Do not overwrite an existing queue'
    assert not OUTPUT.exists(), 'Do not overwrite training output'
    source = DATA / 'manifest.json'
    manifest = json.loads(source.read_text()); source_sha = sha(source)
    groups = {t['task_id']: t['train'] for t in manifest['tasks']}
    chosen = select(groups)
    assert len(groups) == 15 and sum(map(len, groups.values())) == 699
    assert sum(map(len, chosen.values())) == 35
    metadata_hashes, records = {}, []
    for task in manifest['tasks']:
        train, val = set(task['train']), set(task['val'])
        assert train.isdisjoint(val) and len(train) == len(task['train'])
        meta = DATA / task['trajectory_index']; metadata_hashes[str(meta)] = sha(meta)
        assert metadata_hashes[str(meta)] == task['trajectory_index_sha256']
        indexed = {r['sample_id']: r for r in map(json.loads, meta.read_text().splitlines())}
        for sample in chosen[task['task_id']]:
            assert sample in train and sample not in val
            row = indexed[sample]; assert row.get('training_eligible', True)
            key = f"Count/{task['task_id']}/{sample}"
            views = row['video_views']
            order = ['main_view', 'wrist_view', None] if 'wrist_view' in views else ['main_view', 'left_view', 'right_view']
            videos = [str(meta.parent / views[v]['video_path']) if v in views else None for v in order]
            hashes = [views[v]['video_sha256'] if v in views else None for v in order]
            assert videos[0] and all(Path(p).is_file() for p in videos if p)
            assert row['progress_policy'] == 'whole-video-cumulative-visual-delta-v1'
            progress = meta.parent / row['progress_path']
            progress_sha = sha(progress); assert progress_sha == row['progress_sha256']
            provenance = json.loads(progress.with_suffix('.json').read_text())
            assert provenance['protocol_sha256'] == sha(LABEL_SETUP / 'protocol.json')
            assert provenance['video_sha256'] == hashes[0]
            prompt = row.get('task_prompt') or row.get('source_instruction'); assert prompt
            record = dict(key=key, suite='Count', task_id=task['task_id'], sample_id=sample,
                split='train', frames=row['total_frames'], fps=row['fps'], videos=videos,
                video_hashes=hashes, progress_path=str(progress), progress_sha256=progress_sha,
                task_prompt=prompt, output=str(CACHE / key), metadata_sha256=metadata_hashes[str(meta)],
                generate_missing_progress=False)
            progress_values(record)
            records.append(record)
    records.sort(key=lambda r: r['key'])
    assert sha(source) == source_sha
    reference = json.loads((REFERENCE / 'training.config.json').read_text())
    assert reference['run_protocol']['suite'] == 'Order' and reference['epochs'] == 1
    assert reference['base_revision'] == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    # Preserve the validated latest State/Order setup, not the retired four-suite configuration.
    save(ROOT / 'reference-training.config.json', reference)
    shutil.copyfile(source, ROOT / 'count-manifest.snapshot.json')
    shutil.copyfile(LABEL_SETUP / 'protocol.json', ROOT / 'progress-protocol.json')
    with sqlite3.connect(ROOT / 'cache-queue.sqlite') as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE jobs (key TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT DEFAULT "pending", worker INTEGER, samples INTEGER, error TEXT)')
        db.executemany('INSERT INTO jobs(key,payload) VALUES(?,?)', [(r['key'], json.dumps(r)) for r in records])
    save(ROOT / 'subset-manifest.json', dict(status='prepared', seed=42, selected_episodes=35,
        available_episodes=699, task_count=15, minimum_per_task=2,
        tasks={k: dict(available=len(groups[k]), selected=len(v)) for k, v in chosen.items()},
        selection='Two uniformly sampled training episodes per task; five random tasks receive one extra episode',
        records=records, source_manifest=str(source), source_manifest_sha256=source_sha,
        source_metadata_sha256=metadata_hashes, validation_episodes=sum(len(t['val']) for t in manifest['tasks']),
        validation_unchanged=True, existing_visual_labels_verified=True, labels_unchanged=True, created=time.time()))
    save(ROOT / 'authorization.json', dict(user_instruction='Count 抽取35条轨迹，保证每个task至少2条轨迹，开始训练。',
        suite='Count', episodes=35, epochs=1, fps=5, seed=42, gpus=[0,1],
        reference_setup=str(REFERENCE), fresh_original_base=True, old_weights_reused=False,
        automatic_cache_to_training=True, other_tasks_unchanged=True, desktop_automation_created=False))
    print(json.dumps({'selected': 35, 'tasks': {k:len(v) for k,v in chosen.items()},
                      'raw_frames':sum(r['frames'] for r in records), 'labels_verified':True}), flush=True)


if __name__ == '__main__':
    main()
