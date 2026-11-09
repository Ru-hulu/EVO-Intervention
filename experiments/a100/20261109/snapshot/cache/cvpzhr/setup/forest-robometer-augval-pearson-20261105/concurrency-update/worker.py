"""Independent GPU shards; each model sees the identical complete augmented timelines."""
import argparse, json, os, sys, time, traceback
from common import *
class IndexedDataset:
    def __init__(self,data):self.data=data
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return dict(self.data[i],evaluation_index=i)
class IndexedCollator:
    def __init__(self,collator):self.collator=collator
    def __call__(self,rr):
        batch=self.collator(rr);batch['indices']=[r['evaluation_index'] for r in rr];return batch
class RobometerCollator:
    def __init__(self,collator):self.collator=collator
    def __call__(self,rr):return self.collator.encode(rr),rr
def cache(p,rank):
    import torch
    import prepare_static_jitter as aug
    sys.path.insert(0,str(F_CODE))
    from mem_lite.memory.vision import load_vision_encoder
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(0.30)
    aug.CONTRACT=p['contract']
    episodes=assigned(p,rank)
    visual,processor,digest=load_vision_encoder(p['model'],'cuda')
    assert digest==p['history_source']['vision_sha256']
    for n,e in enumerate(episodes,1):
        receipt=aug.build_episode(p,e,visual,processor)
        # Confirm the generated memory rows use exactly the shared evaluation timeline.
        actual=rows(Path(e['output_dir'])/'samples.jsonl');expected=rows(e['timeline'])
        for a,b in zip(actual,expected,strict=True):
            assert all(a[k]==b[k] for k in ('sample_index','source_sample_index','anchor_sample_index',
                       'source_frame_index','frame_index','timestamp_seconds','kind','pt','images'))
        write(ROOT/'receipts'/(e['episode_id']+'.json'),receipt)
        write(ROOT/f'cache-{rank}.json',dict(phase='building',pid=os.getpid(),completed_episodes=n,total_episodes=len(episodes)))
    write(ROOT/f'cache-{rank}.json',dict(phase='completed',pid=os.getpid(),completed_episodes=len(episodes)))
def record(meta,value):
    assert math.isfinite(value) and 0<=value<=1
    return {**{k:meta[k] for k in ('episode_id','suite','task','sample_index','source_sample_index',
              'anchor_sample_index','source_frame_index','frame_index','timestamp_seconds','kind')},
            'target_pt':meta['pt'],'predicted_pt':value}
def evaluate(p,kind,rank):
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(2)
    episodes=assigned(p,rank)
    meta=[r for e in episodes for r in rows(e['timeline'])]
    path=OUT/kind/f'predictions-{rank}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        existing=rows(path)
        assert len(existing)==len(meta)
        write(ROOT/f'{kind}-{rank}.json',dict(phase='completed',samples=len(existing),reused=True))
        return
    state=lambda **kw:write(ROOT/f'{kind}-{rank}.json',dict(pid=os.getpid(),updated_at=now(),**kw))
    state(phase='loading_model',completed_samples=0,total_samples=len(meta))
    if kind=='forest':
        sys.path.insert(0,str(F_CODE))
        from mem_lite.datasets.m3_dataset import M3Dataset,Stage3Collator
        from mem_lite.runtime import load_checkpoint
        data=M3Dataset([e['output_dir'] for e in episodes])
        assert len(data)==len(meta) and data.history_source==p['history_source']
        model,processor,config=load_checkpoint(F_CKPT,device='cuda')
        collator=Stage3Collator(processor,data.history_dim,config['max_sequence_length'],data.history_capacity)
        loader=DataLoader(IndexedDataset(data),batch_size=4,shuffle=False,
            collate_fn=IndexedCollator(collator),num_workers=2,pin_memory=True,multiprocessing_context='spawn')
    else:
        sys.path.insert(0,str(R_CODE))
        sys.path.insert(0,str(R_CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'))
        from robometer_training.model import load_model,load_processor,load_adapter,current_logits,progress_values
        from robometer_training.data import ProgressCollator
        config=read(R_SETUP/'config.json')
        receipt=read(R_CKPT/'complete.json')
        assert receipt['complete'] and receipt['step']==9967,receipt
        assert read(R_CKPT/'trainer_state.json')['global_step']==9967
        processor=load_processor(config['model_path'],min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
        model,_=load_model(config['model_path'],processor)
        load_adapter(model,R_CKPT)
        model.to('cuda').eval()
        data=[r for e in episodes for r in rows(e['robometer_rows'])]
        for a,b in zip(data,meta,strict=True):
            assert a['episode_id']==b['episode_id'] and a['sample_index']==b['sample_index'] and a['pt']==b['pt']
        loader=DataLoader(data,batch_size=8,shuffle=False,collate_fn=RobometerCollator(ProgressCollator(processor,config['max_length'])),
            num_workers=2,pin_memory=True,multiprocessing_context='spawn')
    completed=0;start=time.monotonic();last=start
    with path.with_suffix('.partial').open('w') as stream,torch.inference_mode():
        for batch in loader:
            if kind=='forest':
                inputs={k:v.to('cuda',non_blocking=True) if torch.is_tensor(v) else v for k,v in batch['inputs'].items()}
                values=model(inputs)['pt'].float().cpu().tolist()
                assert batch['indices']==list(range(completed,completed+len(values)))
            else:
                inputs,rr=batch
                inputs={k:v.to('cuda',non_blocking=True) for k,v in inputs.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=current_logits(model,inputs)
                assert torch.isfinite(logits).all()
                values=progress_values(logits).cpu().tolist()
                assert [r['id'] for r in rr]==[r['id'] for r in data[completed:completed+len(values)]]
            for value in values:
                stream.write(json.dumps(record(meta[completed],value),ensure_ascii=False,allow_nan=False)+'\n')
                completed+=1
            if time.monotonic()-last>=15:
                stream.flush()
                state(phase='evaluating',completed_samples=completed,total_samples=len(meta),elapsed_seconds=time.monotonic()-start)
                last=time.monotonic()
    assert completed==len(meta)
    path.with_suffix('.partial').replace(path)
    state(phase='completed',completed_samples=completed,total_samples=len(meta),elapsed_seconds=time.monotonic()-start,
          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['cache','forest','robometer'])
    parser.add_argument('--rank',type=int,required=True);args=parser.parse_args()
    p=plan()
    try:
        if args.kind=='cache':cache(p,args.rank)
        else:evaluate(p,args.kind,args.rank)
    except BaseException as exc:
        write(ROOT/f'{args.kind}-{args.rank}.json',dict(phase='failed',error=repr(exc),traceback=traceback.format_exc(),pid=os.getpid()))
        raise
if __name__=='__main__':main()
