"""Restore isolated inference dependencies; never launch dataset evaluation."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = Path('/cache/cvpzhr/setup/zeroshot-rebuild-20261113/environment')
UV = '/cache/cvpzhr/tools/bootstrap-packages/bin/uv'
PY312 = '/cache/cvpzhr/tools/python/cpython-3.12.16-linux-x86_64-gnu/bin/python3.12'
ENVS = {
    'robometer': Path('/cache/cvpzhr/envs/robometer'),
    'generative': Path('/cache/robot-baselines/envs/procvlm'),
    'vlac': Path('/cache/robot-baselines/envs/vlac'),
}


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def copy_vlac_binary_stack(destination):
    """Independent copies, with no shared mutable site-packages or global edits."""
    assert os.sys.version_info[:2] == (3, 10)
    source = Path('/opt/venv/lib/python3.10/site-packages')
    target = destination / 'lib/python3.10/site-packages'
    stamp = destination / 'binary-stack-copy.json'
    if stamp.exists():
        return
    names = ['torch', 'torchvision', 'torchaudio', 'triton', 'flash-attn']
    torch = importlib.metadata.distribution('torch')
    names += [r.split('==')[0] for r in torch.requires or [] if r.startswith('nvidia-')]
    names = list(dict.fromkeys(names))
    roots, versions = set(), {}
    for name in names:
        d = importlib.metadata.distribution(name)
        versions[name] = d.version
        for f in d.files or []:
            part = Path(str(f)).parts[0]
            if part not in ('..', '/') and (source / part).exists():
                roots.add(part)
    assert versions['torch'] == '2.8.0+cu128'
    assert versions['flash-attn'] == '2.8.3'
    target.mkdir(parents=True, exist_ok=True)
    for name in sorted(roots):
        src, dst = source / name, target / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True, symlinks=False,
                            ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(src, dst)
    save(stamp, {'source': str(source), 'versions': versions, 'independent_copies': True})


def restore_sources():
    import requests
    for name, spec in json.loads((HERE / 'runtime-sources.json').read_text()).items():
        destination = Path(spec['destination'])
        marker = ROOT / f'{name}-source.json'
        if marker.exists() and destination.is_dir():
            assert json.loads(marker.read_text()) == spec
            continue
        archive = ROOT / f"{name}-{spec['revision']}.tar.gz"
        if not archive.exists():
            session = requests.Session(); session.trust_env = False
            url = f"https://codeload.github.com/{spec['repo']}/tar.gz/{spec['revision']}"
            with session.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with archive.open('wb') as sink:
                    for block in response.iter_content(1024 * 1024):
                        sink.write(block)
            session.close()
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == spec['archive_sha256']
        with tarfile.open(archive) as source:
            members = source.getmembers()
            prefix = members[0].name.split('/')[0]
            for member in members:
                parts = Path(member.name).parts
                assert parts and parts[0] == prefix and '..' not in parts and not member.name.startswith('/')
                if len(parts) < 2:
                    continue
                dest = destination.joinpath(*parts[1:])
                assert dest.resolve().is_relative_to(destination.resolve())
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as file:
                        dest.write_bytes(file.read())
                else:
                    raise ValueError('Refuse non-regular archive member: ' + member.name)
        save(marker, spec)


def prepare(name):
    destination = ENVS[name]
    state = {'status': 'installing', 'started_at': time.time(), 'pid': os.getpid(),
             'environment': str(destination), 'evaluation_started': False}
    path = ROOT / f'{name}-status.json'
    save(path, state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
               UV_CACHE_DIR='/cache/cvpzhr/tools/uv-cache', UV_LINK_MODE='copy',
               UV_HTTP_TIMEOUT='180', UV_CONCURRENT_DOWNLOADS='4',
               UV_CONCURRENT_INSTALLS='2', OMP_NUM_THREADS='4')
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy',
                'PYTHONPATH', 'UV_INDEX_URL', 'PIP_INDEX_URL', 'PIP_EXTRA_INDEX_URL']:
        env.pop(key, None)
    try:
        with (ROOT / f'{name}-install.log').open('a') as log:
            def run(args):
                log.write('\n' + ' '.join(args) + '\n'); log.flush()
                subprocess.run(args, env=env, stdout=log, stderr=log, check=True)
            if not (destination / 'bin/python').exists():
                run([UV, 'venv', '--python', '/opt/venv/bin/python' if name == 'vlac' else PY312,
                     str(destination)])
            if name == 'vlac':
                copy_vlac_binary_stack(destination)
            requirements = HERE / f'requirements-{name}.txt'
            command = [UV, 'pip', 'install', '--python', str(destination / 'bin/python'),
                       '--index-url', 'https://mirrors.huaweicloud.com/repository/pypi/simple',
                       '--index-strategy', 'unsafe-best-match', '-r', str(requirements)]
            if name == 'robometer':
                command += ['--extra-index-url', 'https://download.pytorch.org/whl/cu130']
            run(command)
            run([UV, 'pip', 'check', '--python', str(destination / 'bin/python')])
            lock = subprocess.check_output([UV, 'pip', 'freeze', '--python', str(destination / 'bin/python')],
                                           env=env, text=True)
            (ROOT / f'{name}-environment.lock.txt').write_text(lock)
            state.update(status='installed', finished_at=time.time(),
                         requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
                         lock_sha256=hashlib.sha256(lock.encode()).hexdigest())
    except Exception:
        state.update(status='failed', error=traceback.format_exc(), finished_at=time.time())
    save(path, state)
    return state


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'installation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    restore_sources()
    with ThreadPoolExecutor(3) as pool:
        results = dict(zip(ENVS, pool.map(prepare, ENVS)))
    save(ROOT / 'installation-status.json', {'environments': results,
         'status': 'installed' if all(r['status'] == 'installed' for r in results.values()) else 'failed',
         'evaluation_started': False, 'completed_at': time.time()})


if __name__ == '__main__':
    main()
