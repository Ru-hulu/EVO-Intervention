"""Install verified training inputs and retain only copper files 000--057."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tarfile
import time

BASE = Path('/root/dataset')
JOB = BASE / '.migration_jobs/a100_20261009'
SPECS = [
    ('piperx_insert_copper_screw', 'RW-RL-Dataset/piperx_insert_copper_screw', 58),
    ('piperx-bolt-intentional-drop-reinsert-22ep-raw', 'piperx-bolt-intentional-drop-reinsert-22ep-raw', 22),
]
CAMERA_MAP = {
    'observation.images.camera_top': 'observation.images.right_environment_1',
    'observation.images.camera_wrist_left': 'observation.images.left_wrist',
    'observation.images.camera_wrist_right': 'observation.images.right_wrist',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def replace_paths(value, name, destination):
    if isinstance(value, str):
        source_prefix = f'/root/dataset/{name}'
        if value == source_prefix or value.startswith(source_prefix + '/'):
            value = str(BASE / destination) + value[len(source_prefix):]
            if name.startswith('piperx-bolt-'):
                for old, new in CAMERA_MAP.items():
                    value = value.replace('/videos/' + old + '/', '/videos/' + new + '/')
        return value
    if isinstance(value, list):
        return [replace_paths(v, name, destination) for v in value]
    if isinstance(value, dict):
        return {k: replace_paths(v, name, destination) for k, v in value.items()}
    return value


def main():
    manifest = json.loads((JOB / 'transfer_manifest.json').read_text())
    archive = JOB / 'training_inputs.tar.gz'
    assert digest(archive) == (JOB / 'archive.sha256').read_text().split()[0]
    staging = JOB / 'staging'
    staging.mkdir(exist_ok=True)
    expected = {r['target']: r for r in manifest['files']}
    with tarfile.open(archive, 'r:gz') as tf:
        assert set(tf.getnames()) == set(expected)
        for member in tf.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute()
            assert '..' not in Path(member.name).parts
            dest = staging / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as src, dest.open('wb') as out:
                shutil.copyfileobj(src, out)
            assert digest(dest) == expected[member.name]['sha256']
    print('Transferred files verified:', len(expected), flush=True)

    report = {'created_at': time.time(), 'archive_sha256': digest(archive),
              'transferred_files': len(expected), 'datasets': []}
    for name, destination, count in SPECS:
        root = staging / destination
        final = BASE / destination
        assert not final.is_symlink()
        expected_ids = {f'file-{i:03d}' for i in range(count)}
        stats = json.loads((root / 'samples/statistics.json').read_text())
        assert {r['video_id'] for r in stats['sources']} == expected_ids
        assert {p.stem for p in (root / 'labels').glob('*.JSON')} == expected_ids
        assert {p.name.removesuffix('-delta.txt') for p in (root / 'delta').glob('*-delta.txt')} == expected_ids
        for source in stats['sources']:
            stem = source['video_id']
            assert digest(root / 'labels' / f'{stem}.JSON') == source['annotation_sha256']
            assert digest(root / 'delta' / f'{stem}-delta.txt') == source['delta_sha256']
        index = root / 'samples/samples.jsonl'
        old_hash = digest(index)
        pending = index.with_suffix('.jsonl.tmp')
        path_set, ids, max_frames = set(), set(), {}
        semantic_before, semantic_after = hashlib.sha256(), hashlib.sha256()
        n = 0
        with index.open() as src, pending.open('w') as out:
            for line in src:
                record = json.loads(line)
                assert record['video_id'] in expected_ids
                new = replace_paths(record, name, destination)
                before = {k: v for k, v in record.items() if k != 'video_paths'}
                after = {k: v for k, v in new.items() if k != 'video_paths'}
                assert before == after
                semantic_before.update(json.dumps(before, sort_keys=True).encode())
                semantic_after.update(json.dumps(after, sort_keys=True).encode())
                for p in new['video_paths'].values():
                    path = Path(p)
                    if p not in path_set:
                        assert path.is_relative_to(final / 'videos')
                        assert path.is_file() and not any(q.is_symlink() for q in [path, *path.parents])
                    path_set.add(p)
                    max_frames[p] = max(max_frames.get(p, -1), max(f for f in new['frame_indices'] if f is not None))
                ids.add(new['video_id'])
                out.write(json.dumps(new, ensure_ascii=False, separators=(',', ':')) + '\n')
                n += 1
        assert ids == expected_ids and len(path_set) == count * 3
        assert n == stats['summary']['total_samples']
        assert semantic_before.digest() == semantic_after.digest()
        pending.replace(index)
        for p in (root / 'samples').glob('*.json'):
            save(p, replace_paths(json.loads(p.read_text()), name, destination))
        for folder in ['samples', 'labels', 'delta']:
            target = final / folder
            assert not target.exists(), f'Unexpected existing target: {target}'
        report['datasets'].append({
            'dataset': name, 'root': str(final), 'videos': count, 'samples': n,
            'unique_video_paths': len(path_set), 'source_index_sha256': old_hash,
            'adapted_index_sha256': digest(index),
            'non_path_fields_sha256': semantic_after.hexdigest(),
            'max_referenced_frame': max_frames,
        })
        print('Index verified:', name, n, 'samples,', len(path_set), 'video paths', flush=True)

    # Publish only after both indexes and every source checksum pass.
    for name, destination, count in SPECS:
        for folder in ['samples', 'labels', 'delta']:
            (staging / destination / folder).rename(BASE / destination / folder)
    save(JOB / 'index_migration_report.json', report)

    copper = BASE / SPECS[0][1]
    dirs = [copper / 'data/chunk-000'] + sorted((copper / 'videos').glob('*/chunk-000'))
    assert len(dirs) == 4
    keep, remove = [], []
    for directory in dirs:
        suffix = '.parquet' if directory == dirs[0] else '.mp4'
        assert {p.name for p in directory.iterdir()} == {f'file-{i:03d}{suffix}' for i in range(100)}
        for i in range(100):
            p = directory / f'file-{i:03d}{suffix}'
            assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(copper)
            row = {'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
            (keep if i < 58 else remove).append(row)
    assert len(remove) == 168 and len(keep) == 232
    save(JOB / 'delete_058_099_manifest.json', {'files': remove, 'bytes': sum(r['size'] for r in remove)})

    # Rebuild LeRobot metadata from the retained trajectories, including statistics.
    import numpy as np
    import pyarrow.parquet as pq
    meta = copper / 'meta'
    meta_backup = JOB / 'original_copper_meta'
    shutil.copytree(meta, meta_backup)
    episode_path = meta / 'episodes/chunk-000/file-000.parquet'
    table = pq.read_table(episode_path)
    selected = [i for i, row in enumerate(table.select(['episode_index', 'data/file_index']).to_pylist())
                if row['episode_index'] < 58]
    selected_table = table.take(selected)
    assert selected_table['episode_index'].to_pylist() == list(range(58))
    assert selected_table['data/file_index'].to_pylist() == list(range(58))
    arrays = {'observation.state': [], 'action': []}
    frames = 0
    for i in range(58):
        data = pq.read_table(copper / 'data/chunk-000' / f'file-{i:03d}.parquet')
        assert set(data['episode_index'].to_pylist()) == {i}
        assert len(data) == selected_table['length'][i].as_py()
        frames += len(data)
        for key in arrays:
            arrays[key].append(np.array(data[key].to_pylist(), dtype=np.float64))
    new_stats = {}
    for key, chunks in arrays.items():
        a = np.concatenate(chunks)
        new_stats[key] = {'count': [len(a)], 'min': a.min(0).tolist(), 'max': a.max(0).tolist(),
                          'mean': a.mean(0).tolist(), 'std': a.std(0).tolist(),
                          'q01': np.quantile(a, .01, axis=0).tolist(),
                          'q99': np.quantile(a, .99, axis=0).tolist()}
    info = json.loads((meta / 'info.json').read_text())
    info.update(total_episodes=58, total_frames=frames, duration=frames / info['fps'],
                splits={'train': '0:58'},
                data_files_size_in_mb=round(sum(r['size'] for r in keep if r['path'].endswith('.parquet')) / 1024**2, 6),
                video_files_size_in_mb=round(sum(r['size'] for r in keep if r['path'].endswith('.mp4')) / 1024**2, 6))
    pq.write_table(selected_table, str(episode_path) + '.tmp')
    os.replace(str(episode_path) + '.tmp', episode_path)
    save(meta / 'stats.json', new_stats)
    save(meta / 'info.json', info)
    for row in remove:
        p = Path(row['path'])
        assert p.stat().st_size == row['size'] and p.stat().st_mtime_ns == row['mtime_ns']
        p.unlink()
    for row in keep:
        p = Path(row['path'])
        assert p.stat().st_size == row['size'] and p.stat().st_mtime_ns == row['mtime_ns']
    for directory in dirs:
        assert len(list(directory.iterdir())) == 58
    save(JOB / 'cleanup_report.json', {'deleted_files': len(remove), 'freed_bytes': sum(r['size'] for r in remove),
                                     'retained_data_and_video_files': len(keep), 'metadata_episodes': 58,
                                     'metadata_frames': frames, 'original_metadata_backup': str(meta_backup)})
    print('Cleanup verified:', len(remove), 'files deleted;', frames, 'frames retained', flush=True)


if __name__ == '__main__':
    main()
