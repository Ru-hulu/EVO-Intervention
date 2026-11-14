"""Verified native-frame caches for frozen group-2/5 playback maps.

Reuses the baseline's pinned decoder byte-for-byte. Playback order, repeated
frames and truth remain in the frozen map; each required source frame is stored
once. No background preparation or GPU inference is started by this module.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

from . import eval_common
from .prefetch_images import publish
from .prepare_cache import save, sha
from .stream_manifests import digest
from .stream_runtime import validate_entry


@contextmanager
def image_namespace(root):
    """Private single-worker staging; never alter another process's decoder."""
    previous = eval_common.IMAGE_ROOT
    eval_common.IMAGE_ROOT = Path(root)
    try:
        yield
    finally:
        eval_common.IMAGE_ROOT = previous


def signature(row, entry, main_only):
    validate_entry(row, entry)
    issues = row.get('input_issues', [])
    if any(not main_only or i.get('view') == 0 for i in issues):
        raise ValueError('Required source view is blocked; no frame filling is permitted')
    if not row['videos'][0]:
        raise ValueError('Main view is required')
    unique = sorted(set(entry['source_frame_indices']))
    return {'video_hashes': row['video_hashes'][:1] if main_only else row['video_hashes'],
            'frame_indices': unique}


def verify_cache(root, row, expected, main_only):
    proof = json.loads((root/'verification.json').read_text())
    assert proof['input'] == expected and proof['decoded_native_frames'] == row['frames']
    assert proof['lossless'] is True
    views = [0] if main_only else [i for i, path in enumerate(row['videos']) if path]
    names = {f'{frame:08d}-{view}.png' for frame in expected['frame_indices'] for view in views}
    assert set(proof['png_sha256']) == names, 'Missing or unexpected cached source frame'
    for name, value in proof['png_sha256'].items():
        assert sha(root/name) == value, name
    return root


def prepare(row, entry, main_only=False, cache_root=None):
    """May be called by an admitted worker; never launches a GPU process."""
    if type(main_only) is not bool:
        raise ValueError('main_only must be boolean')
    cache_root = Path(cache_root or eval_common.IMAGE_ROOT).resolve()
    expected = signature(row, entry, main_only)
    relative = Path(row['key'])
    assert not relative.is_absolute() and '..' not in relative.parts
    key = digest({'key': row['key'], 'native_frames': row['frames'], 'input': expected})
    # Perturbations preserve the entire original 5-FPS sequence. FPS=5 uses the
    # same exact source indices too; reuse the already verified baseline cache.
    if expected['frame_indices'] == row['frame_indices']:
        destination = cache_root/relative/('fps5-main' if main_only else 'fps5')
    else:
        destination = cache_root/'mapped-native'/key
    if destination.exists():
        # An existing incomplete directory is an error, never permission to
        # replace or remove a consumer's data.
        return verify_cache(destination, row, expected, main_only)
    try:
        import av
    except ModuleNotFoundError as exc:
        if exc.name != 'av':
            raise
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
        environment.pop('PYTHONPATH', None)
        response = subprocess.run([
            '/cache/cvpzhr/envs/forest-labels/bin/python', '-m',
            'experiments.prm_five_groups_20261114.stream_images'],
            input=json.dumps(dict(row=row, entry=entry, main_only=main_only,
                                  cache_root=str(cache_root))),
            text=True, stdout=subprocess.PIPE, check=True,
            cwd=Path(__file__).parents[2], env=environment)
        result = json.loads(response.stdout)
        assert result['path'] == str(destination) and result['decoder'] == 'PyAV 19.0.2'
        return verify_cache(destination, row, expected, main_only)
    assert av.__version__ == '19.0.2', 'Decoder version changed'
    locks = cache_root/'.stream-locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks/f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            return verify_cache(destination, row, expected, main_only)
        staging = cache_root/'.stream-staging'/str(uuid.uuid4())
        try:
            decoded_row = dict(row, frame_indices=expected['frame_indices'])
            with image_namespace(staging):
                root = eval_common.ensure_images(decoded_row, main_only=main_only)
            verify_cache(root, row, expected, main_only)
            # Atomic no-replace publishing also protects against a writer that
            # does not use our advisory lock.
            publish(root, destination)
            verify_cache(destination, row, expected, main_only)
        except BaseException:
            save(staging/'failure.json', {'at': time.time(), 'source_key': row['key'],
                                        'input': expected, 'error': traceback.format_exc()})
            raise
        else:
            shutil.rmtree(staging)
    return destination


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Decoder bridge is CPU-only'
    data = json.load(sys.stdin)
    path = prepare(**data)
    print(json.dumps({'path': str(path), 'decoder': 'PyAV 19.0.2',
                      'verification_sha256': sha(path/'verification.json')}))
