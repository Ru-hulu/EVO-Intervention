import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from mem_lite.runtime import load_checkpoint
from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.periodic_validation import make_validation_loader, validate


def main():
    rank = int(os.environ['RANK'])
    local = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(2)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    output = Path('/cache/cvpzhr/evaluations/forest-s01-sampling-comparison-20261106')
    checkpoint = '/cache/cvpzhr/experiments/forest-s01-decline/01-uniform-20261106/last.pt'
    config = json.loads(Path('/cache/cvpzhr/outputs/forest-s01-critical-sampling-20261106/config.json').read_text())
    baseline = json.loads(Path('/cache/cvpzhr/experiments/forest-s01-decline/01-uniform-20261106/config.json').read_text())
    assert config['validation_sample_files'] == baseline['validation_sample_files']
    for key in ('max_sequence_length', 'max_pixels', 'history_source'):
        assert config[key] == baseline[key], key
    started = time.monotonic()
    model, processor, saved = load_checkpoint(checkpoint, device=f'cuda:{local}')
    dataset = M3Dataset(config['validation_sample_files'], model.history_projection.in_features)
    loader = make_validation_loader(config, processor, dataset, rank, world)
    if rank == 0:
        print(json.dumps({'event': 'validation_started', 'samples': len(dataset), 'checkpoint': checkpoint}), flush=True)
    result = validate(model, loader, torch.device('cuda', local))
    if rank == 0:
        result.update(checkpoint=checkpoint, step=13224, seconds=time.monotonic()-started,
                      comparison='Same validation files, evaluator, batch size and 4-rank partition as weighted run')
        path = output/'uniform-step13224.json'
        assert not path.exists()
        path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
