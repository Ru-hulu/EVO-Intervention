"""Run native four-rank FoREST fresh, then verify its two-epoch checkpoint."""
import math
import os
import time
from .common import *


def main():
    import torch
    import importlib.metadata
    from mem_lite.train_stage3 import train
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    assert int(os.environ['WORLD_SIZE']) == 4
    assert importlib.metadata.version('flash-linear-attention') == '0.3.2'
    assert callable(qwen.chunk_gated_delta_rule) and callable(qwen.FusedRMSNormGated)
    native = read(ROOT/'training.config.json')
    assert native['epochs'] == 2 and native['batch_size']*4*native['gradient_accumulation_steps'] == 8
    started = time.time(); train(native)
    if int(os.environ['RANK']) != 0: return
    if (ROOT/'resource-stop.json').exists():
        save(ROOT/'paused.json', dict(status='resource_paused', checkpoint=str(OUT/'last.pt'), updated=time.time()))
        return
    expected = read(ROOT/'plan.json')['expected_steps']; path = OUT/'epoch-02.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 2 and checkpoint['next_batch'] == 0 and checkpoint['step'] == expected
    assert checkpoint['optimizer']['state'] and checkpoint['scheduler'] is None
    assert len(checkpoint['rng_by_rank']) == 4
    assert checkpoint['config']['training_draws'] == native['training_draws']
    assert checkpoint['config']['data_signature'] == native['data_signature']
    for state in checkpoint['model'].values():
        assert state and all(torch.isfinite(t).all() for t in state.values())
    init = [read(OUT/f'initialization-rank{i}-fresh.json') for i in range(4)]
    assert all(v['fresh_language_lora'] and not v['resume'] and v['world_size'] == 4 for v in init)
    assert len({v['initial_trainable_sha256'] for v in init}) == 1
    logs = [r for r in rows(OUT/'train.jsonl') if r['kind'] == 'train']
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in logs)
    assert any(r['sampling']['history_dropout_draws'] > 0 for r in logs)
    assert any(r['sampling']['near_anomaly_draws'] > 0 for r in logs)
    verify_code(); verify_bindings()
    save(ROOT/'checkpoint-verification.json', dict(status='passed', checkpoint=str(path), checkpoint_sha256=sha(path),
        steps=expected, epochs=2, fresh_official_initialization=True, optimizer_constant_lr_four_rank_rng=True,
        source_data_and_cached_histories_unchanged=True, training_draws_sha256=native['training_draws']['sha256'],
        sampling=read(ROOT/'sampling-verification.json'), peak_reserved_gib=max(max(r['peak_reserved_gib_per_gpu']) for r in logs),
        seconds=time.time()-started, completed=time.time()))


if __name__ == '__main__': main()
