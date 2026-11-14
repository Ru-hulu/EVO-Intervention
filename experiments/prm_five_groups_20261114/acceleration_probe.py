"""Disposable native FoREST forward/backward and two-rank resume probes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time


def tensor_digest(named):
    h = hashlib.sha256()
    for name, p in named:
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def run(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DistributedSampler
    from transformers import AutoProcessor
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
    from mem_lite.models.m3_predictor import M3ProgressPredictor
    from mem_lite.train_stage3 import load_trainable_state, build_scheduler, normalize_gradients
    from contextlib import nullcontext

    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    assert world == (2 if args.ddp_smoke else 1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.cuda.set_per_process_memory_fraction(.22, device)
    if world == 2: dist.init_process_group('nccl')
    assert q.is_fast_path_available == (args.backend == 'accelerated')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    c = ckpt['config']
    assert c['batch_size'] == 1 and c['gradient_accumulation_steps'] == 9
    assert c['model']['gradient_checkpointing'] is True
    data = M3Dataset(c['sample_files'], history_dim=c['model']['history_dim'])
    assert data.signature == c['data_signature']
    processor = AutoProcessor.from_pretrained(c['model_path'], max_pixels=c['max_pixels'], local_files_only=True)
    collate = Stage3Collator(processor, data.history_dim, c['max_sequence_length'], c['history_capacity'])
    model = M3ProgressPredictor(c['model_path'], **c['model']).to(device)
    load_trainable_state(model, ckpt['model'])
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    initial_digest = tensor_digest(named)
    model.train()
    sampler = DistributedSampler(data, num_replicas=2, rank=rank, seed=c['seed'], shuffle=True)
    sampler.set_epoch(ckpt['epoch'])
    order = list(sampler)
    def batch(index):
        row = collate([data[index]])
        return {k:v.to(device) if torch.is_tensor(v) else v for k,v in row['inputs'].items()}, row['target_pt'].to(device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    receipt = dict(backend=args.backend, initial_trainable_sha256=initial_digest,
                   checkpoint_step=ckpt['step'], data_signature=data.signature, config=c,
                   fast_path=q.is_fast_path_available, official_predictions_written=False)
    if not args.ddp_smoke:
        indices = [0, 17, len(data)//4, len(data)//2, len(data)-1, order[ckpt['next_batch']]]
        values = []
        for case,index in enumerate(indices):
            inputs, target = batch(index)
            model.zero_grad(set_to_none=True)
            torch.manual_seed(4242+case);torch.cuda.manual_seed_all(4242+case)
            torch.cuda.synchronize(); started=time.perf_counter()
            result=model(inputs,target);result['loss_sum'].backward()
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            gradients={name:p.grad.detach().cpu().clone() for name,p in named if p.grad is not None}
            assert all(torch.isfinite(g).all() for g in gradients.values())
            values.append(dict(index=index, pt=result['pt'].detach().cpu(),
                               loss=result['loss_sum'].detach().cpu(), gradients=gradients,
                               seconds=elapsed, sequence_length=inputs['input_ids'].shape[-1]))
            print(json.dumps(dict(case=case,index=index,seconds=elapsed,pt=result['pt'].item())),flush=True)
        receipt.update(cases=values,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                       peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        torch.save(receipt,output/f'{args.backend}.pt')
    else:
        parameters=[p for _,p in named]
        optimizer=torch.optim.AdamW(parameters,lr=c['learning_rate'],weight_decay=c['weight_decay'],fused=c['fused_optimizer'])
        scheduler=build_scheduler(optimizer,dict(c),len(sampler))
        optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler'])
        wrapped=DistributedDataParallel(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
        rng=ckpt['rng_by_rank'][rank]
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device)
        random.setstate(rng['python'])
        records=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True); loss_sum=0.0
            torch.cuda.synchronize();started=time.perf_counter()
            for micro in range(9):
                index=order[ckpt['next_batch']+step*9+micro]
                inputs,target=batch(index)
                with wrapped.no_sync() if micro<8 else nullcontext():
                    result=wrapped(inputs,target);result['loss_sum'].backward()
                loss_sum+=float(result['loss_sum'].detach())
            normalize_gradients(parameters,torch.tensor(18.,device=device),2)
            norm=torch.nn.utils.clip_grad_norm_(parameters,c['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();scheduler.step();torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            assert all(torch.isfinite(p).all() for p in parameters)
            records.append(dict(step=ckpt['step']+step+1,seconds=elapsed,loss=loss_sum/9,
                                gradient_norm=float(norm),lr=optimizer.param_groups[0]['lr']))
        assert scheduler.last_epoch==ckpt['step']+2
        receipt.update(status='passed', rank=rank,world_size=world,steps=records,
            optimizer_scheduler_rng_restored=True, next_batch=ckpt['next_batch']+18,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            diagnostic_weights_never_reused=True)
        (output/f'{args.backend}-ddp-rank{rank}.json').write_text(json.dumps(receipt,indent=2)+'\n')
        dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['original','accelerated'],required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--ddp-smoke',action='store_true');run(p.parse_args())
