"""Independent SOC queues, immutable caches, fresh model initialization per suite."""
import hashlib
import json
from pathlib import Path
ROOT=Path('/cache/cvpzhr/setup/soc35-procvlm-robometer-e1-20261114')
CACHE_SETUP=Path('/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114')
CACHE=Path('/cache/cvpzhr/datasets/robometer-procvlm-soc35-5fps-20261114')
OUTPUT=Path('/cache/cvpzhr/outputs/soc35-procvlm-robometer-e1-20261114')
SUITES=['State','Order','Count']
GPUS={'robometer-4b':[0,1],'procvlm-2b':[2,3]}
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
