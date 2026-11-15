import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/cache/cvpzhr/setup/soc-grm-validation-20261115')
DATA = Path('/cache/cvpzhr/datasets/soc-grm-validation-5fps-20261115')
SOURCE = Path('/cache/cvpzhr/setup/soc-disrupted-eval-20261115/inputs.json')
SELECTION = Path('/cache/cvpzhr/setup/soc-two-per-task-perturbations-20261114/selection.json')
TRAIN = Path('/cache/cvpzhr/setup/soc35-grm-finetune-20261115')
TRAIN_SELECTION = Path('/cache/cvpzhr/setup/soc35-grm-caches-20261115/selection.json')
TRAIN_PROTOCOL = TRAIN_SELECTION.parent/'protocol.json'
NATIVE = Path('/cache/cvpzhr/setup/nine-model-zeroshot-cleanval-voc-20261113')
MODELS = ('robobrain25-4b', 'dopamine-3b')
SUITES = ('State', 'Order', 'Count')
EXPECTED = dict(State=22, Order=32, Count=30)


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    temp.replace(path)


def freeze(path, value):
    if Path(path).exists():
        assert read(path) == value, f'Refuse changed frozen artifact: {path}'
    else: save(path, value)
    return sha(path)


def freeze_rows(path, value):
    path = Path(path)
    if path.exists(): assert rows(path) == value, f'Refuse changed input cache: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in value))
        temp.replace(path)
    return sha(path)


def resolve(path, expected, current=Path('/cache/cvpzhr/datasets/ProPriMem'),
            backup=Path('/cache/cvpzhr/datasets/ProPriMem-before-hf-align-20261115')):
    path = Path(path)
    if path.is_file() and sha(path) == expected: return path
    assert path.is_relative_to(current), f'Changed non-dataset source: {path}'
    old = backup/path.relative_to(current)
    assert old.is_file() and sha(old) == expected, f'Original source unavailable: {path}'
    return old


def image_triplet(row, frame):
    return [str(Path(row['image_root'])/f'{frame:08d}-{v}.png') if p else None
            for v, p in enumerate(row['videos'])]


def native_input(row, index, prompt):
    frames = row['frame_indices']
    assert 0 <= index < len(frames)
    first, last, current = (image_triplet(row, frames[i]) for i in (0, -1, index))
    return dict(task=row['task_prompt'], prompt_path=prompt,
                images=[first[0], last[0], *first, *current])


def check_checkpoint_metadata(model, suite, proof, cfg, selection_sha):
    assert proof['status'] == 'passed' and proof['pilot'] is False
    assert proof['epoch'] == 1 and proof['expected_episodes'] == 35
    assert proof['suite'] == suite == cfg['suite'] and cfg['name'] == model
    assert cfg['epochs'] == 1 and cfg['episodes'] == 35 and cfg['training_fps'] == 5
    assert proof['selection_sha256'] == cfg['selection_sha256'] == selection_sha
    assert proof['train_input_sha256'] == cfg['train_sha256']
    assert proof['optimizer_scheduler_two_rank_rng'] and proof['replicas_equal']
    assert proof['fresh_official_initialization'] and cfg['future_endpoint_reference']
    assert proof['expected_samples'] == cfg['samples']
    assert Path(proof['checkpoint']).name == f"checkpoint-{proof['steps']}"


def tensor_digest(items):
    import torch
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
