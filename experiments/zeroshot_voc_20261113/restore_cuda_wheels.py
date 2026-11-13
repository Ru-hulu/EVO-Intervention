"""Download two pinned CUDA wheels on A100, checking official PyTorch hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import quote

import requests

ROOT = Path('/cache/cvpzhr/setup/zeroshot-rebuild-20261113/environment/cu130-wheels')
WHEELS = {
    'torch-2.10.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl':
        '858f0cbcc78d726fea9499eb3464faa98392fa093845a3262209bd226b7844d6',
    'torchvision-0.25.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl':
        'c2f5e38f0cd57a2796e4503c0f13365deba01dbc167ef820f0beec7ca96f5f2e',
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(name, expected):
    target = ROOT / name
    if target.exists() and sha(target) == expected:
        return {'file': name, 'bytes': target.stat().st_size, 'sha256': expected}
    url = 'https://mirrors.aliyun.com/pytorch-wheels/cu130/' + quote(name)
    session = requests.Session()
    session.trust_env = False
    with session.get(url + '?probe=total', headers={'Range': 'bytes=0-0'}, timeout=(10, 30)) as response:
        response.raise_for_status()
        match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
        if response.status_code != 206 or not match:
            raise ValueError('Unexpected size probe response')
        total = int(match.group(1))
    parts = ROOT / (name + '.parts')
    parts.mkdir(exist_ok=True)
    chunk = 8 * 1024 * 1024

    def segment(start):
        end = min(total, start + chunk) - 1
        path = parts / str(start)
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(4):
            try:
                client = requests.Session()
                client.trust_env = False
                with client.get(url + f'?segment={start}-{end}',
                                headers={'Range': f'bytes={start}-{end}'},
                                stream=True, timeout=(10, 90)) as reply:
                    reply.raise_for_status()
                    if reply.status_code != 206 or reply.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
                        raise ValueError('Unexpected partial response')
                    temporary = path.with_suffix('.part')
                    with temporary.open('wb') as sink:
                        for block in reply.iter_content(1024 * 1024):
                            sink.write(block)
                    if temporary.stat().st_size != end - start + 1:
                        raise ValueError('Truncated wheel segment')
                    temporary.replace(path)
                    return path
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)
            finally:
                client.close()

    with ThreadPoolExecutor(8) as pool:
        ordered = list(pool.map(segment, range(0, total, chunk)))
    temporary = target.with_suffix('.assembling')
    with temporary.open('wb') as sink:
        for part in ordered:
            with part.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    sink.write(block)
    if sha(temporary) != expected:
        raise ValueError('Wheel differs from the official PyTorch SHA256')
    temporary.replace(target)
    for part in ordered:
        part.unlink()
    parts.rmdir()
    session.close()
    result = {'file': name, 'bytes': total, 'sha256': expected,
              'downloaded_on_a100': True, 'hash_source': 'official PyTorch cu130 index',
              'mirror': 'https://mirrors.aliyun.com/pytorch-wheels/cu130/'}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    results = [download(name, digest) for name, digest in WHEELS.items()]
    (ROOT / 'verification.json').write_text(json.dumps(results, indent=2) + '\n')
