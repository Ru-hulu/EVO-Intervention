import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/cache/cvpzhr/setup/group5-local-density-20261115')
DATA = Path('/cache/cvpzhr/datasets/group5-local-density-20261115')
SOURCE = Path('/cache/cvpzhr/setup/group5-soc35-memory-fps-20261115/inputs.json')
RESULTS = Path('/cache/cvpzhr/evaluations/group5-soc35-memory-fps-20261115')
MODELS = ('robometer-4b', 'procvlm-2b', 'forest')
SUITES = ('State', 'Order', 'Count')
RATES = (6, 12, 18, 24, 30)
SEED = 42


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    temp.replace(path)


def freeze(path, value):
    path=Path(path)
    if path.exists():assert read(path)==value, f'Refuse changed frozen cache: {path}'
    else:save(path,value)
    return sha(path)


def model_path(model, level, key):
    assert model in MODELS and level in range(1,6) and not Path(key).is_absolute() and '..' not in Path(key).parts
    return DATA/'models'/model/f'level{level}'/(key.replace('/','__')+'.json')


def resolve_source(path, expected, current=Path('/cache/cvpzhr/datasets/ProPriMem'),
                   backup=Path('/cache/cvpzhr/datasets/ProPriMem-before-hf-align-20261115')):
    """Resolve only exact original bytes after the separately authorized HF swap."""
    path=Path(path)
    if path.is_file() and sha(path)==expected:return path
    archived=Path(backup)/path.relative_to(current)
    assert archived.is_file() and sha(archived)==expected, f'Original bytes unavailable: {path}'
    return archived
