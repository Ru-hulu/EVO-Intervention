"""独立模型测速，学习率为0、不保存参数；正式训练重新加载基座。"""
import argparse
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.models.m3_predictor import M3ProgressPredictor

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')


def main(args):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device('cuda',local);torch.cuda.set_device(device);torch.set_num_threads(4)
    if world>1:dist.init_process_group('nccl')
    result=dict(batch_per_gpu=args.batch,world_size=world,gradient_checkpointing=args.checkpointing,
                workers=args.workers,mode=args.mode,learning_rate=0,checkpoint_saved=False)
    try:
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        config=json.loads(Path('/cache/cvpzhr/setup/proprimem-5fps-20261030/train-4gpu.template.json').read_text())
        selection=json.loads((RUN/'selection.json').read_text())
        dataset=M3Dataset(config['sample_files']);assert dataset.signature==selection['dataset_signature']
        processor=AutoProcessor.from_pretrained(config['model_path'],max_pixels=config['max_pixels'],local_files_only=True)
        collator=Stage3Collator(processor,dataset.history_dim,config['max_sequence_length'],dataset.history_capacity)
        config['model']['gradient_checkpointing']=args.checkpointing
        model=M3ProgressPredictor(config['model_path'],**config['model']).to(device)
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=0,weight_decay=config['weight_decay'],fused=True)
        wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True) if world>1 else model
        wrapped.train()
        torch.cuda.reset_peak_memory_stats(device)

        def update(batch):
            data={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch['inputs'].items()}
            target=batch['target_pt'].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output=wrapped(data,target)
            output['loss'].backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
            optimizer.step()
            return output['loss'].detach(),norm.detach()

        if args.mode=='probe':
            sample=dataset[selection['stress_index']]
            batch=collator([sample]*args.batch)
            for _ in range(3):loss,norm=update(batch)
            torch.cuda.synchronize(device)
            result.update(stress_sequence_length=batch['inputs']['input_ids'].shape[1],loss=float(loss),grad_norm=float(norm))
        else:
            subset=Subset(dataset,selection['random_indices'][rank::world])
            loader=DataLoader(subset,batch_size=args.batch,shuffle=False,drop_last=True,collate_fn=collator,
                              num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,
                              multiprocessing_context='spawn' if args.workers else None)
            iterator=iter(loader)
            for _ in range(3):update(next(iterator))
            torch.cuda.synchronize(device)
            if world>1:dist.barrier()
            steps=args.steps or max(4,math.ceil(256/(args.batch*world)))
            for repeat in range(2 if args.replay else 1):
                if repeat:
                    result['initial_pass']=dict(result)
                    iterator=iter(loader)
                    for _ in range(3):next(iterator)
                    torch.cuda.synchronize(device)
                    if world>1:dist.barrier()
                wait_seconds=0.0;started=perf_counter();losses=[];norms=[]
                for _ in range(steps):
                    before=perf_counter();batch=next(iterator);wait_seconds+=perf_counter()-before
                    loss,norm=update(batch);losses.append(loss);norms.append(norm)
                torch.cuda.synchronize(device)
                elapsed=perf_counter()-started
                times=torch.tensor([elapsed,wait_seconds],device=device)
                if world>1:dist.all_reduce(times,op=dist.ReduceOp.MAX)
                result.update(measured_steps=steps,seconds=float(times[0]),samples_per_second=steps*args.batch*world/float(times[0]),
                              loader_wait_seconds=float(times[1]),loss_mean=float(torch.stack(losses).mean()),
                              grad_norm_max=float(torch.stack(norms).max()),shape_replay=bool(repeat))
                if rank==0:print(json.dumps(dict(phase='timing_complete',**result)),flush=True)
            del iterator,loader
            # 真实最大序列再做一次前/反向及优化器状态检查，测峰值而不保存模型。
            stress=collator([dataset[selection['stress_index']]]*args.batch)
            stress_loss,stress_norm=update(stress);torch.cuda.synchronize(device)
            result.update(stress_loss=float(stress_loss),stress_grad_norm=float(stress_norm),
                          stress_sequence_length=stress['inputs']['input_ids'].shape[1])
        peak=torch.tensor([torch.cuda.max_memory_allocated(device)/2**30,torch.cuda.max_memory_reserved(device)/2**30],device=device)
        peaks=[torch.empty_like(peak) for _ in range(world)]
        if world>1:dist.all_gather(peaks,peak)
        else:peaks=[peak]
        result.update(status='passed',peak_allocated_gib=[float(p[0]) for p in peaks],peak_reserved_gib=[float(p[1]) for p in peaks])
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status='oom',error=str(exc)[:700])
        if world>1:raise
    finally:
        if rank==0:
            path=RUN/'benchmarks'/f'{args.name}.json';path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['probe','throughput'],default='throughput')
    p.add_argument('--batch',type=int,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--checkpointing',action='store_true');p.add_argument('--name',required=True);p.add_argument('--steps',type=int)
    p.add_argument('--replay',action='store_true',help='同一批样本第二遍计时，排除首次形状编译开销')
    main(p.parse_args())
