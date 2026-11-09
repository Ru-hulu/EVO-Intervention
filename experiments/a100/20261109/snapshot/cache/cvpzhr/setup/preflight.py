"""Four-rank real-model startup checks; never constructs or steps an optimizer."""
import json
import os
from datetime import timedelta
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.memory.vision import verify_vision_source
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.runtime import StreamingProgressPredictor, to_device
from mem_lite.train_stage3 import normalize_gradients

ROOT=Path('/cache/cvpzhr')
rank=int(os.environ['RANK'])
world=int(os.environ['WORLD_SIZE'])
device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
torch.set_num_threads(4)
dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=device)
try:
    assert world==4 and torch.cuda.is_bf16_supported()
    probe=torch.tensor(float(rank+1),device=device)
    dist.all_reduce(probe)
    assert probe.item()==10.0
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config=json.loads((ROOT/'setup/train-4gpu.json').read_text())
    selected=json.loads((ROOT/'setup/preflight-samples.json').read_text())
    dataset=M3Dataset(selected)
    if rank==0: verify_vision_source(config['model_path'],dataset.history_source)
    dist.barrier()
    processor=AutoProcessor.from_pretrained(config['model_path'],max_pixels=config['max_pixels'],local_files_only=True)
    collator=Stage3Collator(processor,dataset.history_dim,config['max_sequence_length'],dataset.history_capacity)
    model=M3ProgressPredictor(config['model_path'],**config['model']).to(device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    wrapped=DistributedDataParallel(model,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=True)
    model.train()
    first=[i for i,(_,_,previous) in enumerate(dataset.samples) if previous is None]
    last=[i-1 for i in first[1:]]+[len(dataset)-1]
    checks=[]
    for label,indices in [('mixed-empty-history',[first[rank%len(first)]]*2 if rank%2==0 else [last[rank%len(last)]]*2),
                          ('long-history',[last[rank%len(last)]]*2)]:
        wrapped.zero_grad(set_to_none=True)
        batch=collator([dataset[i] for i in indices])
        inputs=to_device(batch['inputs'],device)
        target=batch['target_pt'].to(device)
        torch.cuda.synchronize()
        started=perf_counter()
        result=wrapped(inputs,target)
        assert torch.isfinite(result['pt']).all() and torch.isfinite(result['loss'])
        result['loss_sum'].backward()
        normalize_gradients(parameters,2*world,world)
        gradients=[p.grad for p in parameters if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        norm=torch.nn.utils.clip_grad_norm_(parameters,float('inf'),error_if_nonfinite=True)
        gathered=[torch.empty_like(norm) for _ in range(world)]
        dist.all_gather(gathered,norm)
        torch.testing.assert_close(torch.stack(gathered),norm.expand(world),rtol=1e-5,atol=1e-6)
        torch.cuda.synchronize()
        checks.append({'case':label,'loss':result['loss'].item(),'gradient_norm':norm.item(),
                       'seconds':perf_counter()-started,'valid_history_tokens':inputs['history_valid_mask'].sum(-1).tolist(),
                       'input_tokens':inputs['attention_mask'].sum(-1).tolist()})
    wrapped.zero_grad(set_to_none=True)
    model.eval()
    online=[]
    if rank==0:
        stream=StreamingProgressPredictor(model,processor,dataset.history_source,config['max_sequence_length'])
        for index in range(3):
            sample=dataset[index]
            with torch.no_grad():
                cached=model(to_device(collator([sample])['inputs'],device))['pt'].item()
                predicted=stream(sample['images'],sample['frame_index'],sample['episode_id'],sample['task_prompt'])
            delta=abs(cached-predicted['pt'].item())
            assert delta<0.01
            online.append({'frame':sample['frame_index'],'cached_pt':cached,'online_pt':predicted['pt'].item(),
                           'absolute_difference':delta,'history_length':predicted['history_length']})
        sample=dataset[0]
        reset=stream(sample['images'],sample['frame_index'],'new-episode',sample['task_prompt'])
        assert reset['history_length']==0
    record={'rank':rank,'gpu':torch.cuda.get_device_name(device),'checks':checks,
            'allocated_peak_gib':torch.cuda.max_memory_allocated(device)/2**30,
            'reserved_peak_gib':torch.cuda.max_memory_reserved(device)/2**30}
    records=[None]*world
    dist.all_gather_object(records,record)
    if rank==0:
        report={'status':'PASS','world_size':world,'nccl_all_reduce':True,'ranks':records,
                'online_cache_checks':online,'online_cache_tolerance':0.01,'episode_reset':True,
                'trainable_parameters':sum(p.numel() for p in parameters),
                'parameter_updates':0,'training_started':False}
        (ROOT/'setup/gpu-preflight.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
finally:
    dist.destroy_process_group()
