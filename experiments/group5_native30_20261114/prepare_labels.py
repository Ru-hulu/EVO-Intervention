"""Freeze the authorized native-30-FPS subset and add only four missing labels.

Reuses the audited SOC visual-label worker unchanged. Does not train models,
launch evaluation, alter metadata/splits, or replace existing labels.
"""
import argparse
from collections import Counter
from fractions import Fraction
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT = Path('/cache/cvpzhr/setup/group5-native30-two-per-task-20261114')
CODE = ROOT/'code'
REFERENCE_CODE = Path('/cache/cvpzhr/setup/prm-five-groups-20261114/code')
sys.path.insert(0, str(CODE))
from experiments.proprimem_visual_pt_20261113.relabel import read, save, sha, verify_csv

DATA = Path('/cache/cvpzhr/datasets/ProPriMem')
ORIGINAL = Path('/cache/cvpzhr/setup/proprimem-visual-pt-20261113')
FPS = (30, 24, 18, 12, 6)
VISION = '1cc7c705e96c07fb52bcb1fb32f5e05271cc249fbc619c1128fd4eddbea47aea'


def freeze(path, value):
    if path.exists():
        assert read(path) == value, f'Frozen artifact differs: {path}'
    else:
        save(path, value)


def sampling_indices(frame_count, source_rate, target_rate):
    interval = Fraction(source_rate, target_rate)
    assert interval >= 1 and frame_count > 0
    result = []
    k = 0
    while math.ceil(k * interval) < frame_count:
        result.append(math.ceil(k * interval))
        k += 1
    if result[-1] != frame_count - 1:
        result.append(frame_count - 1)
    return result


def prepare():
    import numpy as np
    candidate = read(ROOT/'candidate-readiness.json')
    picked = sorted((r for r in candidate['episodes'] if Fraction(r['rate']) == 30), key=lambda r: r['key'])
    assert len(picked) == 60 and Counter(r['suite'] for r in picked) == dict(State=16, Order=30, Count=14)
    assert set(Counter((r['suite'], r['task_id']) for r in picked).values()) == {2}
    protocol = read(ORIGINAL/'protocol.json')
    for relative, expected in protocol['source_code_sha256'].items():
        if relative.startswith('mem_lite/'):
            assert sha(CODE/relative) == expected, relative
    assert sha(CODE/'experiments/proprimem_visual_pt_20261113/relabel.py') == read(ORIGINAL/'verification.json')['finalization_code_sha256']
    # Preserve the exact original protocol bytes/hash; actual job placement is
    # recorded separately rather than misrepresenting its original 16 workers.
    original_bytes = (ORIGINAL/'protocol.json').read_bytes()
    if (ROOT/'protocol.json').exists():
        assert (ROOT/'protocol.json').read_bytes() == original_bytes
    else:
        (ROOT/'protocol.json').write_bytes(original_bytes)
    protocol_sha = sha(ROOT/'protocol.json')
    selected_ids = {r['key'] for r in picked}
    candidates = {r['key']: r for r in picked}
    train = read(Path('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114/selection.json'))
    train_ids = {r['episode_id'] for r in train['episodes']}
    assert not train_ids & selected_ids
    records, protected = [], {}
    train_hashes = {s: set() for s in ('State', 'Order', 'Count')}
    for suite in train_hashes:
        manifest = DATA/suite/'manifest.json'
        protected[str(manifest)] = sha(manifest)
        for task in read(manifest)['tasks']:
            index = DATA/suite/task['trajectory_index']
            protected[str(index)] = sha(index)
            for meta in map(json.loads, index.read_text().splitlines()):
                key = f"{suite}/{task['task_id']}/{meta['sample_id']}"
                if key in train_ids:
                    train_hashes[suite].add(meta['video_sha256'])
                if key not in selected_ids:
                    continue
                assert meta['sample_id'] in task['val'] and meta['sample_id'] not in task['train']
                assert meta['fps'] == 30 and meta['total_frames'] == candidates[key]['frames']
                directory = DATA/suite/task['task_path']
                video = directory/meta['video_path']
                assert sha(video) == meta['video_sha256']
                progress = candidates[key]['progress_path']
                newly_found = False
                shared = Path('/cache/cvpzhr/setup/soc-two-per-task-perturbations-20261114/labels')/key/'progress.csv'
                if progress is None and shared.is_file() and shared.with_suffix('.json').is_file():
                    progress = str(shared)
                    newly_found = True
                missing = progress is None
                csv = Path(progress) if progress else directory/'progress'/(video.stem+'.progress.csv')
                provenance = csv.with_suffix('.json')
                if missing:
                    assert not csv.exists() and not provenance.exists(), f'Unexpected existing label: {csv}'
                row = dict(source_id=key, suite=suite, task_id=task['task_id'], sample_id=meta['sample_id'],
                    split='val', video=str(video), video_sha256=meta['video_sha256'],
                    frames=meta['total_frames'], fps=30, task_prompt=meta['task_prompt'],
                    videos=candidates[key]['videos'], csv=str(csv), provenance=str(provenance),
                    generate_missing_progress=missing, reused_newly_found_label=newly_found)
                if meta.get('annotation_path'):
                    annotation = directory/meta['annotation_path']
                    protected[str(annotation)] = sha(annotation)
                for info in meta['video_views'].values():
                    view = directory/info['video_path']
                    assert sha(view) == info['video_sha256']
                    protected[str(view)] = info['video_sha256']
                if not missing:
                    proof = verify_csv(row, protocol_sha)
                    assert proof['vision_sha256'] == VISION
                    protected[str(csv)] = sha(csv)
                    protected[str(provenance)] = sha(provenance)
                records.append(row)
    records.sort(key=lambda r: r['source_id'])
    assert {r['source_id'] for r in records} == selected_ids
    assert all(r['video_sha256'] not in train_hashes[r['suite']] for r in records)
    jobs = [r for r in records if r['generate_missing_progress']]
    assert len(jobs) == 3 and Counter(r['task_id'] for r in jobs) == dict(O15=1, O18=2)
    assert sum(r['reused_newly_found_label'] for r in records) == 1
    freeze(ROOT/'selection.json', dict(schema='group5-native30-selection-v1', seed=42,
        selection='Reuse assessed selection: independent Random(42) per task, sorted eligible held-out IDs; prefer existing identity-verified labels when >=2 are available.',
        episodes=records, counts=dict(State=16, Order=30, Count=14),
        input_fps=list(FPS), common_scoring_fps=6, no_upsampling=True,
        candidate_sha256=sha(ROOT/'candidate-readiness.json'),
        train_key_and_main_video_hash_overlap=False, augmentation='none'))
    freeze(ROOT/'protected-files.json', protected)
    freeze(ROOT/'input-index.json', jobs)
    db = sqlite3.connect(ROOT/'queue.sqlite')
    assert not db.execute("SELECT name FROM sqlite_master WHERE name='jobs'").fetchall()
    db.execute('CREATE TABLE jobs(source_id TEXT PRIMARY KEY,payload TEXT,frames INTEGER,status TEXT,worker_id INTEGER,started_at REAL,finished_at REAL,error TEXT)')
    db.executemany("INSERT INTO jobs(source_id,payload,frames,status) VALUES(?,?,?,'pending')",
                   [(r['source_id'], json.dumps(r), r['frames']) for r in jobs])
    db.commit(); db.close()
    models = {model: dict(regime='suite_fine_tuned_existing', by_suite={},
                         status='locating_existing_checkpoints', retraining_authorized=False)
              for model in ('procvlm-2b', 'robometer-4b')}
    setups = dict(State='forest-state-10pct-legacy-e1-20261114',
        Order='forest-order-35-legacy-e1-20261114', Count='forest-count-35-legacy-e1-20261114')
    forest = {}
    for suite, setup in setups.items():
        status_path = Path('/cache/cvpzhr/setup')/setup/'training-status.json'
        status = read(status_path)
        assert status['status'] == 'completed' and status['final_checkpoint_verified']
        assert sha(status['checkpoint']) == status['checkpoint_sha256']
        forest[suite] = dict(checkpoint=status['checkpoint'], sha256=status['checkpoint_sha256'], epoch=1)
    models['forest'] = dict(regime='suite_fine_tuned', by_suite=forest)
    freeze(ROOT/'model-registry.json', models)
    save(ROOT/'preparation.json', dict(status='passed', selected_episodes=60, existing_labels_verified=57,
        reused_newly_found_labels=1, missing_labels=3, missing_original_frames=sum(r['frames'] for r in jobs),
        code_sha256=sha(__file__), protocol_sha256=protocol_sha, models=models,
        evaluation_started=False, training_started=False))


def verify():
    import numpy as np
    rows = read(ROOT/'selection.json')['episodes']
    protocol_sha = sha(ROOT/'protocol.json')
    for path, digest in read(ROOT/'protected-files.json').items():
        assert sha(path) == digest, f'Protected source changed: {path}'
    resolved, receipts = [], []
    for row in rows:
        proof = verify_csv(row, protocol_sha)
        assert proof['vision_sha256'] == VISION
        values = np.loadtxt(row['csv'], delimiter=',', skiprows=1)[:, 2].tolist()
        resolved.append(dict(row, progress_sha256=sha(row['csv']), provenance_sha256=sha(row['provenance'])))
        receipts.append(dict(key=row['source_id'], frames=row['frames'], generated=row['generate_missing_progress']))
        row['_values'] = values
    freeze(ROOT/'resolved-selection.json', dict(episodes=resolved, selection_sha256=sha(ROOT/'selection.json')))
    stream_counts = {}
    for fps in FPS:
        entries = []
        for row in rows:
            indices = sampling_indices(row['frames'], 30, fps)
            common = list(range(0, row['frames'], 5))
            positions = {f: i for i, f in enumerate(indices)}
            assert set(common) <= positions.keys()
            entries.append(dict(key=row['source_id'], suite=row['suite'], task_id=row['task_id'],
                input_fps=fps, source_frame_indices=indices, timestamps=[i/30 for i in indices],
                target_pt=[row['_values'][i] for i in indices], progress_sha256=sha(row['csv']),
                common_scoring_indices=[positions[i] for i in common],
                common_scoring_native_frames=common, common_scoring_timestamps=[i/30 for i in common]))
        stream_counts[str(fps)] = sum(len(e['source_frame_indices']) for e in entries)
        freeze(ROOT/'streams'/f'fps{fps}.json', dict(episodes=entries, expected_episodes=60,
            expected_samples=stream_counts[str(fps)], common_scoring_fps=6,
            interpolation=False, labels_recomputed_per_fps=False,
            selection_sha256=sha(ROOT/'selection.json')))
    result = dict(status='completed', labels_ready=60, generated_labels=3, reused_newly_found_labels=1,
        generated_frames=sum(r['frames'] for r in rows if r['generate_missing_progress']),
        verified_original_frames=sum(r['frames'] for r in rows), all_csv_values_verified=True,
        existing_labels_sources_splits_annotations_unchanged=True, original_vision_sha256=VISION,
        input_fps=list(FPS), common_scoring_fps=6, predictions_per_model_per_fps=stream_counts,
        planned_prediction_count_all_three_models=3*sum(stream_counts.values()),
        planned_curves=60*5*3, label_files=receipts, evaluation_started=False, training_started=False,
        evaluation_runtime_integration='pending; this completion covers labels, selection, model registry and frame maps only',
        finished=time.time())
    save(ROOT/'verification.json', result)
    return result


def run():
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (ROOT/'launch.json').exists(), 'Never launch duplicate workers'
    prepare()
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
    used = {int(x.split(',')[0]): int(x.split(',')[1]) for x in gpu_rows.splitlines()}
    assert all(used[g] < 1024 for g in (2, 3)), 'GPU 2/3 are occupied; preserve current tasks'
    procs = []
    for gpu in (2, 3):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        with (ROOT/f'worker-{gpu}.log').open('ab', buffering=0) as log:
            proc = subprocess.Popen([sys.executable, '-u', '-m', 'experiments.proprimem_visual_pt_20261113.relabel',
                'worker', '--setup', str(ROOT), '--worker-id', str(gpu)], cwd=CODE, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        procs.append(proc)
    save(ROOT/'launch.json', dict(started=time.time(), controller_pid=os.getpid(),
        workers={str(g): p.pid for g, p in zip((2, 3), procs)}, actual_worker_count=2,
        actual_gpus=[2, 3], protocol_origin=str(ORIGINAL/'protocol.json'),
        unchanged_label_worker_sha256=sha(CODE/'experiments/proprimem_visual_pt_20261113/relabel.py')))
    db = sqlite3.connect(ROOT/'queue.sqlite')
    while True:
        counts = dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status'))
        exits = [p.poll() for p in procs]
        save(ROOT/'status.json', dict(status='running', counts=counts, worker_returncodes=exits, updated=time.time()))
        if all(code is not None for code in exits):
            break
        time.sleep(5)
    assert exits == [0, 0] and counts == {'complete': 3}, list(db.execute('SELECT source_id,error FROM jobs'))
    save(ROOT/'status.json', verify())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'verify'])
    args = parser.parse_args()
    try:
        if args.mode == 'run':
            run()
        else:
            save(ROOT/'status.json', verify())
    except BaseException:
        save(ROOT/'error.json', dict(error=traceback.format_exc(), time=time.time()))
        raise
