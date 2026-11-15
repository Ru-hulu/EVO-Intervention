"""Native FoREST training and final checkpoint verification, no pilot reuse."""
import argparse
import math
import os
import time
from .common import *


def main():
    import torch
    import importlib.metadata
    from mem_lite.train_stage3 import train
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--pilot', action='store_true')
    args = p.parse_args(); cfg = read(args.config); native = copy.deepcopy(cfg['native'])
    rank = int(os.environ['RANK']); assert int(os.environ['WORLD_SIZE']) == 2
    folder = Path(cfg['setup_dir'])/('pilot' if args.pilot else 'formal')
    output = folder/'checkpoints' if args.pilot else Path(cfg['output_dir'])
    assert importlib.metadata.version('flash-linear-attention') == '0.3.2'
    assert callable(qwen.chunk_gated_delta_rule) and callable(qwen.FusedRMSNormGated)
    native['output_dir'] = str(output)
    if args.pilot: native.update(max_steps=cfg['pilot_steps'], log_every_steps=1, save_every_steps=None)
    started = time.time(); train(native)
    if rank != 0: return
    checkpoint = output/('last.pt' if args.pilot else 'epoch-01.pt')
    value = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = cfg['pilot_steps'] if args.pilot else math.ceil(math.ceil(6002/2/cfg['microbatch'])/cfg['accumulation'])
    assert value['step'] == expected
    if not args.pilot: assert value['epoch'] == 1 and value['next_batch'] == 0
    assert value['optimizer']['state'] and value['scheduler'] is None and len(value['rng_by_rank']) == 2
    assert value['config']['data_signature'] == cfg['native']['data_signature']
    assert value['config']['history_source'] == cfg['native']['history_source']
    for module in value['model'].values():
        assert module and all(torch.isfinite(v).all() for v in module.values())
    init = [read(output/f'initialization-rank{i}-fresh.json') for i in (0, 1)]
    assert all(v['fresh_language_lora'] and not v['resume'] for v in init)
    assert init[0]['initial_trainable_sha256'] == init[1]['initial_trainable_sha256']
    logs = [r for r in rows(output/'train.jsonl') if r['kind'] == 'train']
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in logs)
    save(folder/'verification.json', dict(status='passed', pilot=args.pilot, steps=expected,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), epoch=value['epoch'],
        optimizer_constant_lr_two_rank_rng=True, fresh_official_initialization=True,
        expected_samples=6002, expected_episodes=20, selection_sha256=cfg['selection_sha256'],
        peak_reserved_gib=max(max(r['peak_reserved_gib_per_gpu']) for r in logs),
        pilot_step_seconds=[[r['step_seconds'] for r in logs]],
        samples_per_second=6002/(time.time()-started), seconds=time.time()-started, completed=time.time()))


if __name__ == '__main__': main()
