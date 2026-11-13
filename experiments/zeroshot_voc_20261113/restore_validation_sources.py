"""Restore an explicitly selected validation-source catalog directly on A100.

This restores source files only. It never creates PT labels, changes the frozen
split, or starts evaluation. Missing episodes remain explicitly recorded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import threading
import time
from urllib.parse import quote, urlsplit

import requests

from rebuild_official_weights import atomic_json, fetch_one, safe_error, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--signed-urls', type=Path)
    parser.add_argument('--videos-only', action='store_true')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.state_dir / 'download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    catalog = json.loads(args.catalog.read_text())
    rows = [r for r in catalog['files'] if not args.videos_only or r['path'].endswith('.mp4')]
    root = Path(catalog['local_dir'])
    for row in rows:
        relative = PurePosixPath(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert isinstance(row['size'], int) and row['size'] >= 0
        assert row.get('lfs') or row.get('blob_id')
    state = {'status': 'downloading', 'suite': catalog['suite'], 'repo_id': catalog['repo_id'],
             'revision': catalog['revision'], 'catalog_sha256': hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
             'source_files_transit_local_computer': False, 'dataset_evaluation_started': False,
             'validation_expected_episodes': catalog['validation_expected_episodes'],
             'validation_available_episodes': catalog['validation_available_episodes'],
             'validation_missing_episodes': catalog['validation_missing_episodes'],
             'total_files': len(rows), 'total_bytes': sum(r['size'] for r in rows),
             'videos_only': args.videos_only, 'started_at': time.time(), 'files': {}}
    mutex = threading.Lock()
    last_write = [0.0]

    def update(path, force=False, **fields):
        with mutex:
            state['files'].setdefault(path, {}).update(fields)
            if force or time.time() - last_write[0] > 3:
                state['updated_at'] = time.time()
                state['verified_files'] = sum(r.get('status') == 'verified' for r in state['files'].values())
                state['downloaded_bytes'] = sum(r.get('bytes', 0) for r in state['files'].values())
                atomic_json(args.state_dir / 'download-status.json', state)
                last_write[0] = time.time()

    def download(row):
        target = root / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = verify(target, row)
        if digest:
            update(row['path'], True, status='verified', sha256=digest, bytes=row['size'], reused=True)
            return
        if target.exists():
            raise ValueError('Existing source failed pinned HF hash verification: ' + row['path'])
        part = target.with_name(target.name + '.download-part')
        path = quote(row['path'], safe='/')
        stem = f"https://hf-mirror.com/datasets/{catalog['repo_id']}"
        urls = [f"{stem}/resolve/{catalog['revision']}/{path}"]
        if not row.get('lfs'):
            urls.insert(0, f"https://hf-mirror.com/api/resolve-cache/datasets/{catalog['repo_id']}/{catalog['revision']}/{path}")
            urls += [f'{stem}/resolve/main/{path}?download=true', f"{stem}/raw/{catalog['revision']}/{path}"]
        client = requests.Session()
        client.trust_env = False
        try:
            for attempt in range(1, 4):
                candidates = list(urls)
                if args.signed_urls and args.signed_urls.exists():
                    signed = json.loads(args.signed_urls.read_text()).get(row['path'])
                    if signed:
                        parsed = urlsplit(signed)
                        assert parsed.scheme == 'https' and parsed.hostname and (
                            parsed.hostname.endswith('.hf.co') or parsed.hostname.endswith('.huggingface.co'))
                        candidates.insert(0, signed)
                update(row['path'], True, status='downloading', attempt=attempt,
                       bytes=part.stat().st_size if part.exists() else 0)
                try:
                    fetch_one(client, candidates, part, row['size'], lambda n: update(row['path'], bytes=n))
                    digest = verify(part, row)
                    if not digest:
                        part.replace(part.with_name(part.name + f'.invalid-{int(time.time())}'))
                        raise ValueError('Source differs from the pinned official HF hash')
                    part.replace(target)
                    update(row['path'], True, status='verified', sha256=digest, bytes=row['size'])
                    return
                except Exception as error:
                    update(row['path'], True, status='retrying', error=safe_error(error))
                    if attempt == 3:
                        update(row['path'], True, status='failed')
                        raise
                    time.sleep(attempt * 3)
        finally:
            client.close()

    errors = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download, row): row for row in rows}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append({'file': futures[future]['path'], 'error': safe_error(error)})
    state.update(status='failed' if errors else 'selected_source_files_verified',
                 failures=errors, finished_at=time.time(),
                 full_validation_ready=False,
                 reason='Full original split and original dense PT must pass a separate preflight.')
    atomic_json(args.state_dir / 'download-status.json', state)
    print(json.dumps({k: state[k] for k in ('status', 'total_files', 'total_bytes', 'full_validation_ready')}))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
