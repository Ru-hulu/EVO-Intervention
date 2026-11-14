"""Install pinned acceleration wheels into an isolated, read-only-base overlay."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = Path('/cache/cvpzhr/setup/prm-five-groups-20261114')
AUDIT = BASE / 'state-acceleration'
ORIGINAL = Path('/cache/cvpzhr/envs/forest-labels')
ENV = Path('/cache/cvpzhr/envs/forest-labels-accelerated-20261114')
VERSIONS = {'flash-linear-attention': '0.4.2', 'fla-core': '0.4.2'}


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, temporary, size):
    if size < 8 << 20:
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as out:
            while chunk := response.read(2 << 20):
                out.write(chunk)
        return
    # Bounded remote-only ranges avoid a very slow single GitHub connection.
    cached=temporary.parent.parent/'signed-download-url.json'
    if cached.exists() and time.time()-cached.stat().st_mtime<1800:
        record=json.loads(cached.read_text());assert record['size']==size
        direct=record['url']
    else:
        with urllib.request.urlopen(urllib.request.Request(url, headers={'Range':'bytes=0-1023'}), timeout=30) as r:
            assert r.status == 206 and r.headers['Content-Range'] == f'bytes 0-1023/{size}'
            direct = r.geturl()
            r.read()
    print('Downloading verified ranges directly on A100',flush=True)
    ranges = [(i, start, min(size, start+(8 << 20))-1) for i,start in enumerate(range(0,size,8 << 20))]
    def part(spec):
        i, start, end = spec
        dest = temporary.with_name(temporary.name + f'.part{i}')
        if dest.exists() and dest.stat().st_size == end-start+1:
            return
        req = urllib.request.Request(direct, headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=90) as r, dest.open('wb') as out:
            assert r.status == 206 and r.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
            while chunk := r.read(1 << 20):
                out.write(chunk)
        assert dest.stat().st_size == end-start+1
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(part,ranges))
    with temporary.open('wb') as out:
        for i,_,_ in ranges:
            with temporary.with_name(temporary.name + f'.part{i}').open('rb') as stream:
                while chunk := stream.read(2 << 20):out.write(chunk)


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    receipt = AUDIT / 'environment.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['status'] == 'passed'
        print(receipt)
        return
    wheels = AUDIT / 'wheels'
    wheels.mkdir(exist_ok=True)
    if not ENV.exists():
        subprocess.run([str(ORIGINAL / 'bin/python'), '-m', 'venv', str(ENV)], check=True)
    site = ENV / 'lib/python3.12/site-packages'
    # All additions live here; the original training environment is only read.
    (site / 'original-read-only.pth').write_text(str(ORIGINAL / 'lib/python3.12/site-packages') + '\n')
    sources = []
    for name, version in VERSIONS.items():
        data = get_json(f'https://pypi.org/pypi/{name}/{version}/json')
        wheel, = [r for r in data['urls'] if r['packagetype'] == 'bdist_wheel']
        sources.append(dict(name=name, version=version, filename=wheel['filename'],
                            url=wheel['url'], sha256=wheel['digests']['sha256'],size=wheel['size']))
    data = get_json('https://api.github.com/repos/Dao-AILab/causal-conv1d/releases/tags/v1.7.0')
    asset, = [a for a in data['assets'] if a['name'] ==
              'causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl']
    assert asset['digest'].startswith('sha256:')
    sources.append(dict(name='causal-conv1d', version='1.7.0', filename=asset['name'],
                        url=asset['browser_download_url'], sha256=asset['digest'].split(':')[1],size=asset['size']))
    for row in sources:
        path = wheels / row['filename']
        if not path.exists() or digest(path) != row['sha256']:
            tmp = path.with_suffix('.download')
            download(row['url'],tmp,row['size'])
            assert digest(tmp) == row['sha256'], row['name']
            tmp.replace(path)
        print('Verified official wheel:', row['filename'], flush=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2', PIP_DISABLE_PIP_VERSION_CHECK='1')
    subprocess.run([str(ENV / 'bin/python'), '-m', 'pip', 'install', '--no-deps',
                    *[str(wheels / r['filename']) for r in sources]], check=True, env=environment)
    code = '''import json, torch, transformers, triton
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.modules import FusedRMSNormGated
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from importlib.metadata import version
assert torch.__version__ == '2.10.0+cu130'
assert transformers.__version__ == '5.3.0'
assert triton.__version__ == '3.6.0'
assert torch._C._GLIBCXX_USE_CXX11_ABI
print(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__,
 triton=triton.__version__, torch_path=torch.__file__, acceleration_imports_passed=True,
 fla=version('flash-linear-attention'), causal_conv1d=version('causal-conv1d'))))'''
    output = subprocess.check_output([str(ENV / 'bin/python'), '-c', code], env=environment, text=True)
    verification = json.loads(output.strip().splitlines()[-1])
    assert verification['torch_path'].startswith(str(ORIGINAL) + '/')
    subprocess.run([str(ENV / 'bin/python'), '-m', 'pip', 'check'], env=environment, check=True)
    receipt.write_text(json.dumps(dict(status='passed', created=time.time(), environment=str(ENV),
        base_environment=str(ORIGINAL), base_dependencies_unchanged=True, additions=sources,
        verification=verification, gpu_test_passed=False), indent=2) + '\n')
    print(receipt, flush=True)


if __name__ == '__main__':
    main()
