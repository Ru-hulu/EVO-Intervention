"""Isolated real-model checks. Never used as a formal training entry point."""
import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor

from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.m3_inputs import M3Collator
from mem_lite.datasets.sequence import EpisodeWindows, SequenceCollator
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.models.linear_memory import gated_delta_forward
from mem_lite.runtime import StreamingProgressPredictor, load_checkpoint, to_device
from mem_lite.train_stage3 import configure_history, normalize_gradients, save_checkpoint
from mem_lite.memory.vision import verify_vision_source


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def emit(kind, **values):
    print(json.dumps(dict(kind=kind, timestamp=time.time(), **values), ensure_ascii=False), flush=True)


def memory():
    torch.cuda.synchronize()
    return dict(allocated_gib=torch.cuda.memory_allocated()/2**30,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)


def difference(a, b):
    return float((a.float()-b.float()).abs().max())


def functional(model, processor, cfg, output, device):
    files=json.loads(Path('/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030/validation.sample_files.json').read_text())
    data=M3Dataset(files, history_dim=cfg['model']['history_dim'])
    collate=M3Collator(processor, data.history_dim, cfg['max_sequence_length'], data.history_capacity)
    model.eval()
    report={}
    # Confirm native GPU kernels, then test multi-token initial_state and conv carry.
    first=model.qwen.model.language_model.layers[0].linear_attn
    report['kernel']=first.chunk_gated_delta_rule.__module__
    assert report['kernel'].startswith('fla.'), report['kernel']
    hidden=torch.randn(1,29,model.qwen.config.text_config.hidden_size,device=device,dtype=torch.bfloat16)
    valid=torch.ones(1,29,device=device,dtype=torch.bool)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        native=first(hidden)
        whole,conv,recur=gated_delta_forward(first,hidden,valid)
        parts=[]; c=s=None
        for a,b in [(0,5),(5,13),(13,29)]:
            value,c,s=gated_delta_forward(first,hidden[:,a:b],valid[:,a:b],c,s)
            parts.append(value)
        split=torch.cat(parts,1)
        torch.testing.assert_close(whole,native,atol=0.025,rtol=0.05)
        torch.testing.assert_close(split,whole,atol=0.025,rtol=0.05)
        torch.testing.assert_close(s,recur,atol=0.025,rtol=0.05)
        _,same_c,same_s=gated_delta_forward(first,hidden,torch.zeros_like(valid),conv,recur)
        torch.testing.assert_close(same_c,conv,atol=0,rtol=0)
        torch.testing.assert_close(same_s,recur,atol=2e-5,rtol=2e-5)
    report['linear_chunk_max_abs']=difference(split,whole)
    report['native_max_abs']=difference(whole,native)
    initial=recur.detach().requires_grad_()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        values,_,next_state=gated_delta_forward(first,hidden[:,:3],valid[:,:3],conv,initial)
        (values.float().square().mean()+next_state.square().mean()).backward()
    assert initial.grad is not None and initial.grad.isfinite().all() and initial.grad.abs().max()>0
    report['initial_state_gradient_max']=float(initial.grad.abs().max())
    model.zero_grad(set_to_none=True)
    del hidden,native,whole,conv,recur,split,c,s,parts,value,initial,values,next_state,same_c,same_s
    emit('functional_linear',**report)

    # A nonzero gate ensures history consistency actually exercises cross-attention.
    for adapter in model.memory_adapters.values():
        adapter.gate.data.fill_(0.2)
    hook_calls=[]
    def check_kv(module,args,kwargs):
        assert kwargs.get('past_key_values') is None
        hook_calls.append(1)
    handles=[]
    for layer in model.qwen.model.language_model.layers:
        decoder=getattr(layer,'decoder',layer)
        if decoder.layer_type=='full_attention':
            handles.append(decoder.self_attn.register_forward_pre_hook(check_kv,with_kwargs=True))
    by_episode={}
    for i,(ep,_,_) in enumerate(data.samples):
        by_episode.setdefault(ep,[]).append(i)
    chosen=next(v for v in by_episode.values() if len(v)>=24)
    samples=[data[i] for i in chosen[:24]]
    runner=StreamingProgressPredictor(model,processor,data.history_source,cfg['max_sequence_length'])
    cached=[];online=[];shapes=[];state=None
    with torch.no_grad():
        for sample in samples:
            batch=to_device(collate([sample]),device)
            result=model(batch,state=state); state=result['state']
            cached.append(result['pt'].cpu())
            streamed=runner(sample['images'],sample['frame_index'],sample['episode_id'],sample['task_prompt'])
            online.append(streamed['pt'].cpu())
            assert streamed['history_length']==int(batch['history_valid_mask'].sum())
            shapes.append(tuple((i,tuple(c.shape),tuple(s.shape)) for i,(c,s) in state.layers.items()))
        a,b=torch.cat(cached),torch.cat(online)
        torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
        assert all(shape==shapes[0] for shape in shapes)
        assert len(state.layers)==24
        report['cached_online_max_abs_pt']=difference(a,b)
        report['functional_frames']=len(samples)
        first_sample=samples[0]
        reset=runner(first_sample['images'],first_sample['frame_index'],'preflight-new-episode',first_sample['task_prompt'])
        torch.testing.assert_close(reset['pt'].cpu(),cached[0],atol=0,rtol=0)
        runner.reset()
        reset=runner(first_sample['images'],first_sample['frame_index'],first_sample['episode_id'],first_sample['task_prompt'])
        torch.testing.assert_close(reset['pt'].cpu(),cached[0],atol=0,rtol=0)
        batches=[to_device(collate([sample]),device) for sample in samples[:8]]
        window=model(batches)
        torch.testing.assert_close(window['pt'].cpu(),a[:8],atol=0,rtol=0)
        altered=model(batches[:4]+list(reversed(batches[4:])))
        torch.testing.assert_close(window['pt'][:4],altered['pt'][:4],atol=0,rtol=0)
    report.update(full_attention_past_always_none=bool(hook_calls),linear_state_layers=24,
                  fixed_state_size=True,episode_reset=True,explicit_reset=True,
                  sequential_window_equal=True,future_frame_isolation=True,**memory())
    for handle in handles: handle.remove()
    write(output/'functional.json',report)
    emit('functional_pass',**report)


def capacity(model,processor,cfg,output,device,rank,world):
    files=json.loads(Path(cfg['sample_files']).read_text())
    data=M3Dataset(files,history_dim=cfg['model']['history_dim'])
    configure_history(cfg,data)
    cfg['data_signature']=data.signature
    cfg['sample_files']=files
    sequences={}
    for i,(ep,_,_) in enumerate(data.samples): sequences.setdefault(ep,[]).append(i)
    longest=sorted(sequences.values(),key=len,reverse=True)[:4*world]
    selected=longest[4*rank:4*(rank+1)]
    dataset=EpisodeWindows(data)
    collate=SequenceCollator(processor,data.history_dim,cfg['max_sequence_length'],data.history_capacity)
    model.train()
    wrapped=DDP(model,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=True) if world>1 else model
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=True)
    state=None; records=[]; count=torch.zeros((),device=device); losses=torch.zeros((),device=device)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for w in range(2):
        indices=[[seq[len(seq)-24+w*8+t] for seq in selected] for t in range(8)]
        spec=dict(indices=indices,reset=w==0,group_end=w==1)
        batch=collate(dataset[spec])
        frames=[to_device(frame,device) for frame in batch['inputs']]
        targets=batch['target_pt'].to(device)
        torch.cuda.synchronize();tick=time.perf_counter()
        from contextlib import nullcontext
        with wrapped.no_sync() if world>1 and w==0 else nullcontext():
            result=wrapped(frames,targets,state=state)
            assert torch.isfinite(result['loss'])
            result['loss_sum'].backward()
        state=result['state'].detach()
        count+=result['num_samples'];losses+=result['loss_sum'].detach()
        rec=dict(rank=rank,window=w,loss=float(result['loss']),samples=int(result['num_samples']),
                 max_tokens=max(f['input_ids'].shape[1] for f in frames),
                 max_history=max(int(f['history_valid_mask'].sum(1).max()) for f in frames),
                 seconds=time.perf_counter()-tick,**memory())
        records.append(rec);emit('backward_window',**rec)
        del result,frames,targets,batch
    if world>1:
        dist.all_reduce(count);dist.all_reduce(losses)
    normalize_gradients(parameters,count,world)
    grad=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'],error_if_nonfinite=True)
    assert torch.isfinite(grad)
    # Exactly one disposable optimizer update validates AdamW allocation and checkpoint IO.
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    report=dict(rank=rank,world_size=world,batch_per_gpu=4,sequence_window=8,windows=2,
                gradient_checkpointing=cfg['model']['gradient_checkpointing'],
                valid_frames=int(count),loss=float(losses/count),grad_norm_before_clip=float(grad),
                optimizer_updates=1,formal_training=False,seconds=time.perf_counter()-started,
                windows_detail=records,**memory())
    checkdir=output/'checkpoint-smoke'
    if rank==0: checkdir.mkdir(exist_ok=True)
    if world>1: dist.barrier()
    save_checkpoint(model,optimizer,cfg,checkdir,epoch=0,next_batch=0,step=1)
    if world>1: dist.barrier()
    # Only single GPU reloads; multi GPU verifies gradients/parameters synchronize.
    if world>1:
        checksum=torch.stack([model.progress_head.weight.float().sum(),next(iter(model.memory_adapters.values())).gate.float().sum()])
        gathered=[torch.empty_like(checksum) for _ in range(world)]
        dist.all_gather(gathered,checksum)
        for other in gathered: torch.testing.assert_close(checksum,other,atol=0,rtol=0)
        report['rank_parameter_checksums_equal']=True
    else:
        model.eval()
        test=to_device(M3Collator(processor,data.history_dim,cfg['max_sequence_length'],data.history_capacity)([data[selected[0][0]]]),device)
        with torch.no_grad(): expected=model(test)['pt'].cpu()
        del model,wrapped,optimizer,parameters,state
        gc.collect();torch.cuda.empty_cache()
        restored,_,_=load_checkpoint(checkdir/'last.pt',device=str(device))
        with torch.no_grad(): actual=restored(test)['pt'].cpu()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        report['checkpoint_reload_max_abs_pt']=difference(actual,expected)
    write(output/f'rank-{rank}.json',report)
    if world>1: dist.barrier()
    if rank==0:
        (checkdir/'last.pt').unlink()
        report['temporary_checkpoint_removed']=True
        emit('capacity_pass',**report)
        write(output/'summary.json',report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['functional','capacity'])
    parser.add_argument('--output',required=True);parser.add_argument('--checkpointing',action='store_true')
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'))
    local=int(os.getenv('LOCAL_RANK','0'));device=torch.device('cuda',local);torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl')
    try:
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        cfg=json.loads(Path('mem_lite/configs/a100_crossattn_train.json').read_text())
        cfg['model']['gradient_checkpointing']=args.checkpointing
        cfg['output_dir']=str(output)
        verify_vision_source(cfg['model_path'],cfg['history_source'])
        processor=AutoProcessor.from_pretrained(cfg['model_path'],max_pixels=cfg['max_pixels'],local_files_only=True)
        model=M3ProgressPredictor(cfg['model_path'],**cfg['model']).to(device)
        emit('model_loaded',mode=args.mode,rank=rank,**memory())
        if args.mode=='functional': functional(model,processor,cfg,output,device)
        else: capacity(model,processor,cfg,output,device,rank,world)
    except BaseException as error:
        failure=dict(rank=rank,type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        write(output/f'failure-rank-{rank}.json',failure)
        emit('failure',**failure)
        raise
    finally:
        if world>1 and dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
