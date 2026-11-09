"""Download recovery views directly from pinned Hugging Face official CDN URLs."""
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import av

JOB = Path(__file__).resolve().parent
DATASET = Path('/cache/cvpzhr/datasets/flipcup-recovery-progress-20261105')

def now():
    return datetime.now(timezone.utc).isoformat()

def save(name, data):
    target = JOB / name
    tmp = target.with_suffix(target.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(target)

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def error_text(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return f'HTTP {exc.code}'
    return f'{type(exc).__name__}: {str(exc).split("?")[0][:200]}'

def download(row):
    path = DATASET / 'videos' / row['relative_path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == row['size'] and digest(path) == row['sha256']:
        return {'path': row['relative_path'], 'bytes': row['size'], 'action': 'already_valid'}
    partial = path.with_suffix('.mp4.part')
    for attempt in range(4):
        try:
            value = hashlib.sha256()
            size = 0
            request = urllib.request.Request(row['download_url'], headers={'User-Agent': 'flipcup-view-download/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as stream:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(block)
                    value.update(block)
                    size += len(block)
            if size != row['size'] or value.hexdigest() != row['sha256']:
                raise ValueError('Downloaded size or SHA256 does not match pinned Hugging Face manifest')
            partial.replace(path)
            return {'path': row['relative_path'], 'bytes': size, 'action': 'downloaded'}
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(error_text(exc)) from None
            print(json.dumps({'retry': row['relative_path'], 'attempt': attempt + 1, 'error': error_text(exc)}), flush=True)
            time.sleep(2 ** attempt * 3)

def inspect(row):
    path = DATASET / 'videos' / row['relative_path']
    sha = digest(path)
    if path.stat().st_size != row['size'] or sha != row['sha256']:
        raise ValueError(f'Invalid file: {row["relative_path"]}')
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if not stream.frames or not stream.average_rate:
            raise ValueError(f'Missing frame count/rate: {row["relative_path"]}')
        return {'path': row['relative_path'], 'bytes': row['size'], 'sha256': sha,
                'frames': stream.frames, 'fps': str(stream.average_rate),
                'width': stream.width, 'height': stream.height}

def main():
    manifest = json.loads((JOB / 'download_manifest.json').read_text())
    rows = [row for row in manifest['files'] if 'download_url' in row]
    state = {'status': 'downloading', 'pid': os.getpid(), 'started_at': now(),
             'repo': manifest['repo'], 'revision': manifest['revision'],
             'dataset_root': str(DATASET), 'download_files': len(rows),
             'download_bytes': sum(row['size'] for row in rows),
             'completed_files': 0, 'completed_bytes': 0, 'errors': []}
    save('status.json', state)
    print(json.dumps(state), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                state['completed_files'] += 1
                state['completed_bytes'] += result['bytes']
                print(json.dumps(result), flush=True)
            except Exception as exc:
                state['errors'].append({'path': row['relative_path'], 'error': error_text(exc)})
            state['updated_at'] = now()
            save('status.json', state)
    if state['errors']:
        state.update(status='failed', finished_at=now())
        save('status.json', state)
        raise RuntimeError('Some downloads failed; see status.json')
    state.update(status='validating', updated_at=now())
    save('status.json', state)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            verified = list(pool.map(inspect, manifest['files']))
        lookup = {row['path']: row for row in verified}
        for i in range(35):
            values = [lookup[f'{view}/episode_{i:06d}.mp4'] for view in ['main_view', 'left_view', 'right_view']]
            if len({(row['frames'], row['fps']) for row in values}) != 1:
                raise ValueError(f'View frame count/rate mismatch: episode_{i:06d}')
        report = {'passed': True, 'repo': manifest['repo'], 'revision': manifest['revision'],
                  'trajectory_count': 35, 'video_count': len(verified),
                  'total_bytes': sum(row['bytes'] for row in verified),
                  'main_view_frames': sum(row['frames'] for row in verified if row['path'].startswith('main_view/')),
                  'checked_at': now(), 'files': verified}
        save('verification.json', report)
        state.update(status='completed', finished_at=now(), verified_videos=len(verified))
    except Exception as exc:
        state.update(status='validation_failed', finished_at=now())
        state['errors'].append({'error': error_text(exc)})
        save('status.json', state)
        raise
    save('status.json', state)
    print(json.dumps(state), flush=True)

if __name__ == '__main__':
    main()
