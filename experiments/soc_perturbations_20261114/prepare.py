"""Prepare source images and static/rewind PT mappings; never run model evaluation."""
import argparse
from collections import Counter
import csv
import fcntl
from fractions import Fraction
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid

from experiments.prm_five_groups_20261114 import eval_common
from experiments.prm_five_groups_20261114.prepare_cache import save, sha, progress_values
from experiments.prm_five_groups_20261114.stream_images import image_namespace, verify_cache
from experiments.prm_five_groups_20261114.prefetch_images import publish
from experiments.prm_five_groups_20261114.stream_manifests import digest, freeze, playback_map

ROOT = Path('/cache/cvpzhr/setup/soc-two-per-task-perturbations-20261114')
CACHE = Path('/cache/cvpzhr/datasets/soc-two-per-task-perturbations-5fps-20261114')
DATA = Path('/cache/cvpzhr/datasets/ProPriMem')
OLD = Path('/cache/cvpzhr/setup/prm-five-groups-20261114/evaluation/group1-manifest.json')
OLD_IMAGES = eval_common.IMAGE_ROOT
LABEL_PROTOCOL = Path('/cache/cvpzhr/setup/proprimem-visual-pt-20261113/protocol.json')
VISION_SHA = '1cc7c705e96c07fb52bcb1fb32f5e05271cc249fbc619c1128fd4eddbea47aea'
SUITES = ('State', 'Order', 'Count')
EXCLUDED = 'Order/O03/swap-blocks_episode_001342'
SEEDS = (42, 43, 44)


def read(path):
    return json.loads(Path(path).read_text())


def choose(tasks):
    """One seed42 RNG; fixed suite/task/key order, select before looking at PT."""
    rng = random.Random(42)
    result = []
    for suite in SUITES:
        for task in sorted(tasks[suite], key=lambda t: t['task_id']):
            eligible = sorted(k for k in task['val'] if f"{suite}/{task['task_id']}/{k}" != EXCLUDED)
            assert len(eligible) >= 2 and not set(task['train']) & set(task['val'])
            for sample in sorted(rng.sample(eligible, 2)):
                result.append((suite, task['task_id'], sample))
    return result


def select():
    import av
    manifests = {s: read(DATA/s/'manifest.json') for s in SUITES}
    tasks = {s: manifests[s]['tasks'] for s in SUITES}
    selected = choose(tasks)
    old = {r['key']: r for r in read(OLD)['episodes']}
    source_hashes = {str(DATA/s/'manifest.json'): sha(DATA/s/'manifest.json') for s in SUITES}
    rows, inventory = [], []
    for suite, tid, sample in selected:
        task = next(t for t in tasks[suite] if t['task_id'] == tid)
        index = DATA/suite/task['trajectory_index']
        source_hashes[str(index)] = sha(index)
        assert sha(index) == task['trajectory_index_sha256']
        metadata = next(json.loads(line) for line in index.read_text().splitlines()
                        if json.loads(line)['sample_id'] == sample)
        views = metadata['video_views']
        names = ['main_view', 'wrist_view', None] if 'wrist_view' in views else ['main_view', 'left_view', 'right_view']
        videos = [str(index.parent/views[n]['video_path']) if n in views else None for n in names]
        hashes = [views[n]['video_sha256'] if n in views else None for n in names]
        rates = []
        for path, h in zip(videos, hashes):
            if not path:
                continue
            assert sha(path) == h, path
            source_hashes[path] = h
            with av.open(path) as video:
                stream = video.streams.video[0]
                assert not stream.frames or stream.frames == metadata['total_frames'], path
                rates.append(stream.average_rate)
        assert len(set(rates)) == 1
        key = f'{suite}/{tid}/{sample}'
        progress = index.parent/(metadata.get('progress_path') or
                                 f"progress/episode_{metadata['episode_index']:06d}.progress.csv")
        overlay = not progress.exists()
        if overlay:
            progress = ROOT/'labels'/key/'progress.csv'
        else:
            source_hashes[str(progress)] = sha(progress)
            if metadata.get('progress_sha256'):
                assert sha(progress) == metadata['progress_sha256']
            provenance = progress.with_suffix('.json')
            assert provenance.exists(), progress
            source_hashes[str(provenance)] = sha(provenance)
        annotation = index.parent/metadata['annotation_path'] if metadata.get('annotation_path') else None
        if annotation:
            source_hashes[str(annotation)] = sha(annotation)
            assert sha(annotation) == metadata['annotation_sha256']
        indices = eval_common.sampling_indices(metadata['total_frames'], rates[0], 5)
        row = dict(key=key, suite=suite, task_id=tid, task_name=task['task_name'], sample_id=sample,
                   split='val', frames=metadata['total_frames'], fps=metadata['fps'],
                   source_rate=str(rates[0]), input_fps=5, videos=videos, video_hashes=hashes,
                   frame_indices=indices, timestamps=[i/float(rates[0]) for i in indices],
                   progress_path=str(progress), progress_sha256=None if overlay else sha(progress),
                   progress_overlay=overlay, task_prompt=metadata.get('task_prompt') or metadata['source_instruction'],
                   metadata_path=str(index), metadata_sha256=sha(index), input_issues=[],
                   annotation_path=str(annotation) if annotation else None,
                   annotation_sha256=sha(annotation) if annotation else None,
                   image_root=str(CACHE/'images'/key/'fps5'),
                   missing_view_policy='RGB black 64x64 placeholder; never fill short views')
        if key in old:
            for field in ('videos', 'video_hashes', 'frames', 'frame_indices', 'progress_path', 'progress_sha256'):
                assert row[field] == old[key][field], (key, field)
        assert len(indices) >= 18
        rows.append(row)
    for s in SUITES:
        for t in sorted(tasks[s], key=lambda t: t['task_id']):
            inventory.append(dict(suite=s, task_id=t['task_id'], task_name=t['task_name'],
                train=list(t['train']), val=list(t['val']),
                selected=[r['sample_id'] for r in rows if r['suite'] == s and r['task_id'] == t['task_id']]))
    assert Counter(r['suite'] for r in rows) == {'State': 22, 'Order': 32, 'Count': 30}
    protocol = read(LABEL_PROTOCOL)
    freeze(ROOT/'visual-pt-protocol.json', protocol)
    selection = dict(schema='soc-two-per-task-v1', selection_seed=42, perturbation_seeds=list(SEEDS),
        suite_order=list(SUITES), excluded={EXCLUDED: 'User-excluded short left view: 435 versus 436 frames'},
        tasks=inventory, source_hashes=source_hashes, episodes=rows,
        original_label_protocol_sha256=sha(LABEL_PROTOCOL),
        missing_pt=[r['key'] for r in rows if r['progress_overlay']], evaluation_started=False)
    freeze(ROOT/'selection.json', selection)
    save(ROOT/'status.json', dict(status='selected', episodes=len(rows), tasks=len(inventory),
         missing_pt=len(selection['missing_pt']), evaluation_started=False, updated=time.time()))
    return dict(episodes=len(rows), tasks=len(inventory), missing_pt=selection['missing_pt'])


def images():
    """Hardlink verified existing pixels, decode only new sources in private staging."""
    receipts = []
    for row in read(ROOT/'selection.json')['episodes']:
        expected = dict(video_hashes=row['video_hashes'], frame_indices=row['frame_indices'])
        destination = Path(row['image_root'])
        if not destination.exists():
            staging = CACHE/'.staging'/str(uuid.uuid4())
            previous = OLD_IMAGES/row['key']/'fps5'
            try:
                if previous.exists():
                    verify_cache(previous, row, expected, False)
                    staging.mkdir(parents=True)
                    for p in previous.iterdir():
                        if p.suffix == '.png' or p.name == 'verification.json':
                            os.link(p, staging/p.name)
                    verify_cache(staging, row, expected, False)
                    publish(staging, destination)
                else:
                    with image_namespace(staging):
                        created = eval_common.ensure_images(row)
                    verify_cache(created, row, expected, False)
                    publish(created, destination)
                if staging.exists():
                    shutil.rmtree(staging)
            except BaseException:
                # Preserve any failed staging for inspection.
                raise
        verify_cache(destination, row, expected, False)
        receipts.append(dict(key=row['key'], image_root=str(destination),
                             verification_sha256=sha(destination/'verification.json')))
        save(ROOT/'image-status.json', dict(status='preparing', completed=len(receipts), total=84,
                                           updated=time.time(), last_key=row['key']))
    freeze(ROOT/'image-verification.json', dict(status='passed', episodes=receipts, decoder='PyAV 19.0.2'))
    save(ROOT/'image-status.json', dict(status='completed', completed=84, total=84, updated=time.time()))
    return dict(status='completed', episodes=84)


def labels():
    """Run only after a physical GPU has been independently observed idle."""
    import torch
    from experiments.recover_validation_20261114.controller import idle_gpus
    from experiments.recover_validation_20261114.common import host_sample, safe
    from mem_lite.datasets.prepare_proprimem import whole_video_progress, write_progress
    from mem_lite.datasets.prepare_progress import extract_deltas
    from mem_lite.memory.vision import load_vision_encoder
    gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    resource = host_sample()
    assert gpu in idle_gpus() and safe(resource, gpu, resource['oom_kill']), 'GPU is occupied or unsafe'
    # Recovery's queue has no unlaunched work; a new worker cannot race for this GPU.
    recovery = read('/cache/cvpzhr/setup/recover-five-validation-20261114/status.json')
    assert not recovery.get('pending')
    assert gpu not in [v['gpu'] for v in recovery.get('active', {}).values()]
    freeze(ROOT/'label-resource-admission.json', dict(gpu=gpu, resources=resource, recovery=recovery))
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.25)
    protocol = read(ROOT/'visual-pt-protocol.json')
    visual, processor, digest = load_vision_encoder(protocol['model_path'], 'cuda')
    assert digest == VISION_SHA and not visual.training and not any(p.requires_grad for p in visual.parameters())
    assert str(next(visual.parameters()).dtype) == protocol['frozen_vision_dtype']
    freeze(ROOT/'label-initialization.json', dict(vision_sha256=digest, model_revision=protocol['model_revision'],
        dtype=str(next(visual.parameters()).dtype), attention='sdpa', gpu=gpu,
        language_model_loaded=False, trainable_parameters=0))
    for row in read(ROOT/'selection.json')['episodes']:
        if not row['progress_overlay']:
            continue
        target = Path(row['progress_path'])
        if target.with_suffix('.json').exists():
            assert sha(target) == read(target.with_suffix('.json'))['csv_sha256']
            continue
        assert not target.exists(), 'Preserve unverified preexisting label'
        assert sha(row['videos'][0]) == row['video_hashes'][0]
        r = host_sample()
        assert safe(r, gpu, resource['oom_kill']), 'Resource pressure; preserve completed labels'
        save(ROOT/'label-status.json', dict(status='running', key=row['key'], updated=time.time(), gpu=gpu))
        delta, info = extract_deltas(row['videos'][0], visual, processor,
                                    batch_size=protocol['batch_size'], max_pixels=protocol['max_pixels'])
        assert len(delta) == row['frames'] and abs(info['fps']-row['fps']) < 1e-3
        pt = whole_video_progress(delta)
        write_progress(target, range(len(pt)), delta, pt)
        freeze(target.with_suffix('.json'), dict(policy=protocol['policy'], source_id=row['key'],
            video_sha256=row['video_hashes'][0], csv_sha256=sha(target), frame_count=len(pt),
            model_revision=protocol['model_revision'], vision_sha256=digest,
            protocol_sha256=sha(ROOT/'visual-pt-protocol.json'), all_original_frames=True, annotations_used=False))
        print(json.dumps(dict(completed=row['key'], frames=len(pt))), flush=True)
    save(ROOT/'label-status.json', dict(status='completed', updated=time.time(), gpu=gpu))


def build():
    selection = read(ROOT/'selection.json')
    rows = []
    for source in selection['episodes']:
        row = dict(source, progress_sha256=sha(source['progress_path']))
        truth = progress_values(row)
        row['target_pt'] = [truth[i] for i in row['frame_indices']]
        rows.append(row)
    baseline = dict(schema='soc-perturbation-source-v1', episodes=rows, expected_episodes=84,
                    selection_sha256=sha(ROOT/'selection.json'), input_fps=5)
    freeze(CACHE/'inputs.json', baseline)
    manifests = {}
    for variant in ('static_only', 'rewind_only'):
        for seed in SEEDS:
            condition = f'{variant}-seed{seed}'
            entries = [playback_map(r, variant, seed) for r in rows]
            for row, entry in zip(rows, entries):
                path = CACHE/'playback'/condition/row['key']/'frames.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name('frames.csv.tmp')
                with temporary.open('w', newline='') as out:
                    w = csv.writer(out)
                    w.writerow(['step', 'playback_seconds', 'source_frame_index', 'source_seconds', 'pt',
                                'main_image', 'left_image', 'right_image'])
                    for i, (frame, t, st, pt) in enumerate(zip(entry['source_frame_indices'],
                            entry['playback_timestamps'], entry['source_timestamps'], entry['target_pt'])):
                        paths = [str(Path(row['image_root'])/f'{frame:08d}-{v}.png') if row['videos'][v] else '' for v in range(3)]
                        w.writerow([i, t, frame, st, pt, *paths])
                if path.exists():
                    assert sha(path) == sha(temporary), 'Frozen playback CSV changed'
                    temporary.unlink()
                else:
                    temporary.replace(path)
                entry['playback_csv'] = str(path)
                entry['playback_csv_sha256'] = sha(path)
                entry['map_sha256'] = digest({k: v for k, v in entry.items() if k != 'map_sha256'})
            path = CACHE/'manifests'/f'{condition}.json'
            value = dict(condition=condition, input_fps=5, seed=seed, episodes=entries,
                         expected_episodes=84, expected_samples=sum(len(e['target_pt']) for e in entries),
                         baseline_manifest_sha256=sha(CACHE/'inputs.json'))
            manifests[str(path)] = freeze(path, value)
    freeze(CACHE/'preparation.json', dict(status='prepared_pending_independent_verification',
        episodes=84, tasks=42, versions=504, by_suite=dict(Counter(r['suite'] for r in rows)),
        manifests=manifests, inputs_sha256=sha(CACHE/'inputs.json'),
        evaluation_started=False, memory_must_follow_playback_order=True,
        missing_view_policy='Empty CSV view path means original RGB black 64x64 placeholder',
        model_specific_hidden_states_precomputed=False))
    return dict(episodes=84, versions=504, evaluation_started=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['select', 'images', 'labels', 'build'])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/f'{args.mode}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(globals()[args.mode](), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
