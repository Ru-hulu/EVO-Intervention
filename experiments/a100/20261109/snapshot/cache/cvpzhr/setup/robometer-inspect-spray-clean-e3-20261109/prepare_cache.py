"""Build unaugmented Robometer caches for inspect-spray-bottles only."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

SOURCE = Path('/cache/cvpzhr/setup/robometer-inspect-spray-clean-e3-20261109/source')
OUTPUT = Path('/cache/cvpzhr/datasets/robometer-inspect-spray-5fps-h7-current3-clean-20261109')
SETUP = Path('/cache/cvpzhr/setup/robometer-inspect-spray-clean-e3-20261109')
MODEL = '/cache/cvpzhr/models/Robometer-4B'
EXPECTED = {'train': (25, 7416), 'validation': (7, 1919)}
TASK_COUNT = 1
POLICY = 'none'
WINDOW = 8
VIEWS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def relabel_replay(rows, originals):
    prepared = []
    for i, row in enumerate(rows):
        row = dict(row)
        row['forest_reference_pt'] = row['pt']
        assert row.get('kind', 'original') == 'original', 'Clean cache cannot contain augmented observations'
        prepared.append(row)
    return prepared

def check_source(summary, rows, originals, split):
    assert len(rows) == summary['num_samples']
    episode = summary['episode_id']
    assert summary['source']['task'] == 'inspect-spray-bottles'
    assert summary['source']['split'] == split
    original_count = 0
    for i, row in enumerate(rows):
        assert 0 <= row['pt'] <= 1, (episode, i, 'PT range')
        assert i == row.get('sample_index', i)
        if i:
            assert row['frame_index'] > rows[i - 1]['frame_index'], (episode, i, 'timeline order')
        kind = row.get('kind', 'original')
        j = row.get('source_sample_index', i)
        assert row.get('source_frame_index', row['frame_index']) == originals[j]['frame_index']
        if kind == 'original':
            assert j == original_count, (episode, i, 'original observations reordered/dropped')
            assert row['pt'] == originals[j]['pt'], (episode, i, 'original label changed')
            original_count += 1
        else:
            assert split == 'train' and kind in ('stationary', 'reverse', 'forward')
            anchor = row['anchor_sample_index']
            expected_index = anchor if kind == 'stationary' else j
            assert row['pt'] == originals[expected_index]['pt'], (episode, i, 'PT/source frame mismatch')
        for view in VIEWS:
            assert row['images'][view] == originals[j]['images'][view]
    assert original_count == len(originals)

def make_records(summary, rows, images, target, split):
    episode = summary['episode_id']
    prepared = []
    for i, row in enumerate(rows):
        past = [max(0, j) for j in range(i - 7, i)]
        indices = past + [i, i, i]
        views = ['cam_high'] * 8 + ['cam_left_wrist', 'cam_right_wrist']
        selected = [images[j][view] for j, view in zip(indices, views, strict=True)]
        record = dict(
            id=f'{episode}/{i:08d}', episode_id=episode,
            source_id=summary['source']['source_id'],
            suite=summary['source']['suite'], task_id=summary['source']['task'],
            sample_index=i, frame_index=row['frame_index'],
            source_frame_index=row.get('source_frame_index', row['frame_index']),
            source_sample_index=row.get('source_sample_index', i),
            anchor_sample_index=row.get('anchor_sample_index', i),
            kind=row.get('kind', 'original'),
            timestamp_seconds=row.get('timestamp_seconds', row['frame_index'] / summary['source']['fps']),
            task=summary['task_prompt'], pt=row['pt'], image=selected,
            forest_reference_pt=row['forest_reference_pt'], pt_policy=POLICY,
            input_contract='history7-main-current3-v1', input_views=views,
            missing_current_views=[view for view in VIEWS if row['images'][view] is None],
            history_sample_indices=past, observation_indices=indices,
            history_frame_indices=[rows[j]['frame_index'] for j in past],
            history_source_frame_indices=[rows[j].get('source_frame_index', rows[j]['frame_index']) for j in past],
            pixel_cache=str(target), window_indices=indices)
        assert len(selected) == 10 and max(indices) <= i and indices[-3:] == [i, i, i]
        assert selected[-3:] == [images[i][view] for view in VIEWS]
        if i: assert max(past) == i - 1
        prepared.append(record)
    return prepared

def initialize():
    global torch, processor, collator
    import torch
    from robometer_training.model import load_processor
    from robometer_training.data import ProgressCollator
    torch.set_num_threads(1)
    processor = load_processor(MODEL, min_pixels=4096, max_pixels=65536)
    collator = ProgressCollator(processor, 4096)

def compare_inputs(records, chosen):
    from robometer_training.data import _episode_cache
    selected = [records[i] for i in chosen]
    cached = collator.encode(selected, supervised=True)
    online = collator.encode([{k: v for k, v in row.items() if k != 'pixel_cache'}
                              for row in selected], supervised=True)
    assert set(cached) == set(online), (set(cached), set(online))
    for key in cached:
        a, b = cached[key], online[key]
        if key == 'pixel_values':
            a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
        assert torch.equal(a, b), (selected[0]['episode_id'], chosen, key, 'cached/online mismatch')
    _episode_cache.cache_clear()
    return len(chosen)

def build_episode(job):
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save_file
    from robometer_training.model import progress_batch
    source, split = Path(job[0]), job[1]
    source_hash = sha(source / 'samples.jsonl')
    summary = json.loads((source / 'summary.json').read_text())
    rows = read_rows(source / 'samples.jsonl')
    episode = summary['episode_id']
    original_path = Path(summary['source']['base_dir']) / 'samples.jsonl'
    originals = read_rows(original_path)
    rows = relabel_replay(rows, originals)
    check_source(summary, rows, originals, split)
    for view in VIEWS:
        assert all(row['images'][view] is not None for row in rows), (episode, view, 'Missing view')
    image_root = OUTPUT / 'images' / split / episode
    linked, images = {}, []
    for row in rows:
        current = {}
        for view in VIEWS:
            if row['images'][view] is None:
                current[view] = str(OUTPUT / 'images' / '_missing_rgb64.png')
                linked['missing:' + view] = current[view]
                continue
            rel = Path(row['images'][view])
            assert not rel.is_absolute() and '..' not in rel.parts
            origin, dest = source / rel, image_root / rel
            if str(origin) not in linked:
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    assert os.path.samefile(origin, dest), dest
                else:
                    os.link(origin, dest)
                linked[str(origin)] = str(dest)
            current[view] = linked[str(origin)]
        images.append(current)

    target = OUTPUT / 'pixels' / split / (episode + '.safetensors')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with safe_open(target, framework='pt', device='cpu') as stored:
            assert stored.metadata()['source_samples_sha256'] == source_hash
            assert stored.metadata()['input_contract'] == 'history7-main-current3-v1'
            assert stored.metadata()['augmentation'] == (POLICY if split == 'train' else 'none')
            for view in VIEWS:
                assert stored.get_slice('pixels_' + view).get_shape()[0] == len(rows)
    else:
        tensors, first_frames = {}, {}
        for view in VIEWS:
            pixel_batches, grid_batches = [], []
            for start in range(0, len(images), 16):
                frames = []
                for item in images[start:start + 16]:
                    with Image.open(item[view]) as im:
                        frames.append(im.convert('RGB'))
                if start == 0:
                    first_frames[view] = frames[0]
                result = processor.image_processor(images=frames, return_tensors='pt')
                grids = result['image_grid_thw']
                assert torch.equal(grids, grids[:1].expand_as(grids)), (episode, view)
                patches = int(grids[0].prod())
                pixel_batches.append(result['pixel_values'].reshape(len(frames), patches, -1).to(torch.bfloat16))
                grid_batches.append(grids)
            grids = torch.cat(grid_batches)
            assert torch.equal(grids, grids[:1].expand_as(grids)), (episode, view)
            tensors['pixels_' + view] = torch.cat(pixel_batches)
            tensors['grids_' + view] = grids
            del pixel_batches, grid_batches
        frames = [first_frames['cam_high']] * 8 + [first_frames['cam_left_wrist'], first_frames['cam_right_wrist']]
        canonical = dict(progress_batch(processor, [(summary['task_prompt'], frames)]))
        template = {'input_' + k: v[0].contiguous() for k, v in canonical.items()
                    if k not in ('pixel_values', 'image_grid_thw')}
        expected = torch.cat([tensors['pixels_cam_high'][0]] * 8 +
                             [tensors['pixels_cam_left_wrist'][0], tensors['pixels_cam_right_wrist'][0]])
        assert torch.equal(canonical['pixel_values'].to(torch.bfloat16), expected)
        prog_id = processor.tokenizer.convert_tokens_to_ids('<|prog_token|>')
        assert int((canonical['input_ids'] == prog_id).sum()) == 8
        assert canonical['image_grid_thw'].shape[0] == 10
        assert canonical['input_ids'].shape[1] <= 4096
        tensors.update(template)
        partial = target.with_suffix('.partial')
        save_file(tensors, str(partial), metadata={
            'format': 'robometer-bf16-multiview-pixels-v2', 'source_samples_sha256': source_hash,
            'augmentation': POLICY if split == 'train' else 'none',
            'window_size': '8', 'history_frames': '7', 'current_views': ','.join(VIEWS),
            'input_contract': 'history7-main-current3-v1'})
        partial.replace(target)
        del tensors
    prepared = make_records(summary, rows, images, target, split)
    chosen = {0, min(1, len(rows)-1), len(rows) // 2, len(rows) - 1}
    for kind in ('stationary', 'reverse', 'forward'):
        match = next((i for i, row in enumerate(rows) if row.get('kind') == kind), None)
        if match is not None: chosen.add(match)
    checked = compare_inputs(prepared, sorted(chosen))
    assert sha(source / 'samples.jsonl') == source_hash
    epfile = OUTPUT / 'episodes' / split / (episode + '.jsonl')
    epfile.parent.mkdir(parents=True, exist_ok=True)
    partial = epfile.with_suffix('.partial')
    with partial.open('w') as stream:
        for row in prepared:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    partial.replace(epfile)
    return {
        'episode_id': episode, 'source_id': summary['source']['source_id'],
        'suite': summary['source']['suite'], 'task': summary['source']['task'],
        'split': split, 'samples': len(rows), 'kinds': dict(Counter(r.get('kind', 'original') for r in rows)),
        'unique_images': len(linked), 'missing_views': [view for view in VIEWS if rows[0]['images'][view] is None],
        'pixel_bytes': target.stat().st_size,
        'pixel_cache': str(target), 'episode_jsonl': str(epfile),
        'source_samples_sha256': source_hash, 'original_samples_sha256': sha(original_path),
        'relabelled_observations': sum(r['pt'] != r['forest_reference_pt'] for r in rows),
        'cached_online_examples_checked': checked}

def mixed_task_check(infos):
    initialize()
    selected = {}
    for info in infos:
        if info['task'] not in selected:
            rows = read_rows(info['episode_jsonl'])
            selected[info['task']] = rows[len(rows) // 2]
    records = list(selected.values())
    for start in range(0, len(records), 8):
        batch = records[start:start + 8]
        compare_inputs(batch, list(range(len(batch))))
    return {'tasks': len(records), 'examples': len(records), 'all_input_tensors_equal': True}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    SETUP.mkdir(exist_ok=True)
    from PIL import Image
    blank = OUTPUT / 'images' / '_missing_rgb64.png'
    blank.parent.mkdir(parents=True, exist_ok=True)
    if not blank.exists():
        Image.new('RGB', (64, 64), (0, 0, 0)).save(blank)
    state = {'phase': 'preflight', 'started_at': time.time(), 'completed_episodes': 0,
             'completed_samples': 0, 'workers': args.workers, 'gpu_used': False,
             'training_started': False}
    def update(**values):
        state.update(values)
        write_json(SETUP / 'status.json', state)
        write_json(OUTPUT / 'status.json', state)
    try:
        jobs = []
        source_hashes = {}
        split_ids = {}
        for split, (expected_episodes, expected_samples) in EXPECTED.items():
            path = SOURCE / (split + '.sample_files.json')
            source_hashes[str(path)] = sha(path)
            paths = json.loads(path.read_text())
            assert len(paths) == expected_episodes
            ids = [json.loads((Path(p) / 'summary.json').read_text())['episode_id'] for p in paths]
            assert len(set(ids)) == len(ids)
            split_ids[split] = set(ids)
            jobs.extend((p, split) for p in paths)
        assert not split_ids['train'] & split_ids['validation']
        write_json(SETUP / 'plan.json', dict(
            source=str(SOURCE), output=str(OUTPUT), model=MODEL,
            initialization='not selected; cache uses processor only and does not load model weights',
            sample_fps=5, window_size=WINDOW, view='history-main-current-three', input_contract='history7-main-current3-v1', history_frames=7, images_per_sample=10, current_views=list(VIEWS),
            train_augmentation='none; original frames and original PT only',
            augmentation_policy=POLICY,
            validation_augmentation='none', workers=args.workers, cuda_visible_devices='',
            expected=EXPECTED, source_list_sha256=source_hashes,
            image_storage='independent hardlinks',
            training_authorized=False, automatic_training_start=False))
        update(phase='building', total_episodes=len(jobs))
        infos = {}
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize) as pool:
            futures = {pool.submit(build_episode, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    info = future.result()
                except BaseException as exc:
                    update(phase='failed', error=repr(exc), failed_job=job, traceback=traceback.format_exc())
                    for pending in futures:
                        pending.cancel()
                    raise
                infos[job] = info
                state['completed_episodes'] += 1
                state['completed_samples'] += info['samples']
                if state['completed_episodes'] % 10 == 0:
                    update(last_episode=info['episode_id'], elapsed_seconds=time.time() - state['started_at'])
                    print(json.dumps(state), flush=True)
        update(phase='verifying', elapsed_seconds=time.time() - state['started_at'])
        ordered = [infos[job] for job in jobs]
        manifest = dict(
            format='robometer-proprimem-current3-pixels-v3', source=str(SOURCE),
            model_path=MODEL, sample_fps=5, window_size=WINDOW, view='history-main-current-three', input_contract='history7-main-current3-v1', history_frames=7, images_per_sample=10, current_views=list(VIEWS),
            min_pixels=4096, max_pixels=65536, max_length=4096,
            pixel_dtype='bfloat16', padding='repeat_first_frame_on_left',
            missing_view_policy='all three views required; all five tasks have complete views',
            forest_features_used=False, training_started=False, splits={},
            augmentation=POLICY, validation_augmentation='none', automatic_training_start=False)
        for split, (nepisodes, nsamples) in EXPECTED.items():
            selected = [info for info in ordered if info['split'] == split]
            assert len(selected) == nepisodes
            assert sum(x['samples'] for x in selected) == nsamples
            assert len({x['task'] for x in selected}) == TASK_COUNT
            target = OUTPUT / (split + '.jsonl')
            partial = target.with_suffix('.partial')
            import shutil
            with partial.open('wb') as stream:
                for info in selected:
                    with Path(info['episode_jsonl']).open('rb') as source:
                        shutil.copyfileobj(source, stream)
            partial.replace(target)
            kinds = Counter()
            suites = {}
            for info in selected:
                kinds.update(info['kinds'])
                suite = suites.setdefault(info['suite'], {'episodes': 0, 'samples': 0, 'tasks': set()})
                suite['episodes'] += 1; suite['samples'] += info['samples']; suite['tasks'].add(info['task'])
            for suite in suites.values():
                suite['tasks'] = len(suite['tasks'])
            manifest['splits'][split] = dict(samples=nsamples, episode_count=nepisodes,
                episodes=selected, tasks=TASK_COUNT, kinds=dict(kinds), suites=suites, sha256=sha(target))
        train_kinds = manifest['splits']['train']['kinds']
        assert train_kinds == {'original': 7416}
        assert manifest['splits']['validation']['kinds'] == {'original': 1919}
        mixed = mixed_task_check(ordered)
        assert mixed['tasks'] == TASK_COUNT
        assert all(sha(p) == digest for p, digest in source_hashes.items())
        checks = dict(all_pt_equal_to_displayed_source_frame=True, no_augmentation=True,
                      no_inserted_observations=True,
                      relabelled_observations=sum(x['relabelled_observations'] for x in ordered),
                      original_observations_preserved=True, all_windows_causal=True,
                      episode_boundaries_preserved=True, train_validation_disjoint=True,
                      all_episode_cached_online_inputs_equal=True,
                      cached_online_examples=sum(x['cached_online_examples_checked'] for x in ordered),
                      mixed_task_inputs=mixed,
                      source_lists_unchanged=True, image_links_independent_of_forest_paths=True,
                      current_three_views_synchronized=True,
                      seven_past_main_views=True, ten_images_eight_progress_tokens=True)
        manifest.update(status='complete', checks=checks, completed_at=time.time(),
                        elapsed_seconds=time.time() - state['started_at'])
        write_json(OUTPUT / 'manifest.json', manifest)
        report = dict(status='completed', output=str(OUTPUT), checks=checks,
                      splits={s: {k: v for k, v in data.items() if k != 'episodes'}
                              for s, data in manifest['splits'].items()},
                      pixel_bytes=sum(x['pixel_bytes'] for x in ordered),
                      elapsed_seconds=manifest['elapsed_seconds'], training_started=False)
        write_json(OUTPUT / 'report.json', report)
        write_json(SETUP / 'report.json', report)
        update(phase='completed', completed_at=time.time(), elapsed_seconds=manifest['elapsed_seconds'])
        print(json.dumps(report), flush=True)
    except BaseException as exc:
        update(phase='failed', error=repr(exc), traceback=traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
