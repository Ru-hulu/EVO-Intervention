"""GPU0/1 official-model queue; FoREST is owned by the other workspace."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/cache/cvpzhr/setup/recover-five-finetune-20261114')
CACHE_SETUP = Path('/cache/cvpzhr/setup/recover-six-model-caches-20261114')
CACHE = Path('/cache/cvpzhr/datasets/recover-six-model-caches-5fps-20261114')
OUTPUT = Path('/cache/cvpzhr/outputs/recover-five-finetune-20261114')
ORDER = ['robometer-4b', 'procvlm-2b', 'robobrain25-4b', 'dopamine-3b']
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def tensor_digest(items):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
