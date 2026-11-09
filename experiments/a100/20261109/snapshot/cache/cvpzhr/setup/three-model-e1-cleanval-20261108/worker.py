
"""Fresh inference on original validation observations using the existing clean caches."""
import argparse, gc, json, math, os, sys, time, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'code/forest'),str(ROOT/'code/robometer'),str(ROOT/'code/robometer-upstream'),str(ROOT/'code/robobrain')]
from common import read, rows, write, now
class Indexed:
    def __init__(self,data):self.data=data
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i],i
class Collate:
    def __init__(self,kind,base):self.kind,self.base=kind,base
    def __call__(self,rr):
        examples,indices=zip(*rr)
        if self.kind=='forest':batch=self.base(list(examples))['inputs']
        else:batch=self.base.encode(list(examples),supervised=False)
        return batch,list(indices)
def main(kind,rank):
    import torch
    from torch.utils.data import DataLoader
    from transformers import set_seed
    run=read(ROOT/'run.json');output=Path(run['output_dir'])/kind;output.mkdir(exist_ok=True)
    progress=ROOT/f'{kind}-{rank}.json'
    state=dict(status='loading',kind=kind,rank=rank,pid=os.getpid(),started_at=now(),completed_samples=0,
               gpu=os.environ['CUDA_VISIBLE_DEVICES'],invalid_generations=0)
    write(progress,state)
    try:
        torch.set_num_threads(2);set_seed(42)
        torch.cuda.set_per_process_memory_fraction(0.85)
        episode_ids=run['shards'][rank]['episode_ids']
        selected=[e for e in run['episodes'] if e['episode_id'] in set(episode_ids)]
        meta=rows(ROOT/f'meta-{rank}.jsonl')
        if kind=='forest':
            from mem_lite.datasets.m3_dataset import M3Dataset,Stage3Collator
            from mem_lite.runtime import load_checkpoint
            data=M3Dataset([e['source_dir'] for e in selected])
            assert data.history_source==run['history_source']
            model,processor,cfg=load_checkpoint(run['checkpoints']['forest']['path'],device='cuda')
            base=Stage3Collator(processor,data.history_dim,cfg['max_sequence_length'],data.history_capacity)
        elif kind=='robometer':
            from robometer_training.model import load_model,load_processor,load_adapter,current_logits,progress_values
            from robometer_training.data import ProgressCollator
            cfg=read(ROOT/'robometer-config.json')
            processor=load_processor(cfg['model_path'],min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'])
            model,_=load_model(cfg['model_path'],processor)
            load_adapter(model,run['checkpoints'][kind]['path']);model.to('cuda').eval()
            data=rows(ROOT/f'robometer-{rank}.jsonl')
            base=ProgressCollator(processor,cfg['max_length'])
        else:
            from grm import Dataset,Collator,load_model,load_processor,generate,parse_score
            cfg=read(ROOT/'robobrain-config.json')
            all_data=Dataset(cfg['data_dir'],'validation')
            wanted=set(episode_ids)
            indices=[i for i,r in enumerate(all_data.rows) if r['episode_id'] in wanted]
            data=[all_data[i] for i in indices]
            processor=load_processor(cfg['model_path'],cfg['min_pixels'],cfg['max_pixels'])
            model=load_model(cfg['model_path'],adapter=run['checkpoints'][kind]['path']).to('cuda').eval()
            base=Collator(processor,cfg['max_length'])
        assert len(data)==len(meta)==run['shards'][rank]['samples']
        if kind!='forest':
            for r,m in zip(data,meta,strict=True):
                assert r['episode_id']==m['episode_id'] and r['sample_index']==m['sample_index'] and r['pt']==m['pt']
        collator=Collate(kind,base)
        loader=DataLoader(Indexed(data),batch_size=run['batch_sizes'][kind],shuffle=False,
            collate_fn=collator,num_workers=4,pin_memory=True,prefetch_factor=2,multiprocessing_context='spawn')
        state.update(status='running',total_samples=len(meta),batch_size=run['batch_sizes'][kind],loaded_at=now())
        write(progress,state)
        def infer(cpu,indices):
            gpu=None
            try:
                gpu={k:v.to('cuda',non_blocking=True) if torch.is_tensor(v) else v for k,v in cpu.items()}
                if kind=='forest':values=model(gpu)['pt'].float().cpu().tolist();answers=[None]*len(values)
                elif kind=='robometer':
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=current_logits(model,gpu)
                    assert torch.isfinite(logits).all()
                    values=progress_values(logits).cpu().tolist();answers=[None]*len(values)
                else:
                    assert 'labels' not in gpu and 'target_pt' not in gpu
                    answers=generate(model,processor,gpu,cfg['max_new_tokens'])
                    values=[]
                    for answer in answers:
                        try:values.append(parse_score(answer))
                        except ValueError:values.append(None)
                return list(zip(indices,values,answers,strict=True))
            except torch.cuda.OutOfMemoryError:
                gpu=None;gc.collect();torch.cuda.empty_cache()
                if len(indices)==1:raise
            mid=len(indices)//2
            state['oom_splits']=state.get('oom_splits',0)+1
            write(progress,state)
            result=[]
            for part in [indices[:mid],indices[mid:]]:
                cpu_part,ii=collator([(data[i],i) for i in part])
                result.extend(infer(cpu_part,ii))
            return result
        path=output/f'predictions-{rank}.jsonl';partial=path.with_suffix('.partial')
        assert not path.exists() and not partial.exists(),'Refuse to overwrite predictions'
        start=time.monotonic();last=start;completed=0
        with partial.open('x') as f,torch.inference_mode():
            for cpu,indices in loader:
                assert indices==list(range(completed,completed+len(indices)))
                predictions=infer(cpu,indices)
                for i,value,answer in predictions:
                    record={k:v for k,v in meta[i].items() if k!='pt'}
                    if value is not None:
                        assert math.isfinite(value)
                        assert -1<=value<=1 if kind=='robobrain25' else 0<=value<=1
                    else:state['invalid_generations']+=1
                    record.update(target_pt=meta[i]['pt'],predicted_pt=value)
                    if answer is not None:record['raw_response']=answer
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    completed+=1
                elapsed=time.monotonic()-start
                if completed==len(indices) or time.monotonic()-last>=15 or completed==len(meta):
                    f.flush();last=time.monotonic()
                    state.update(completed_samples=completed,elapsed_seconds=elapsed,
                        samples_per_second=completed/elapsed,updated_at=now(),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                    write(progress,state)
            f.flush()
        assert completed==len(meta)
        partial.replace(path)
        if state['invalid_generations']:raise RuntimeError('Malformed native GRM scores; predictions preserved and full metrics withheld')
        state.update(status='completed',finished_at=now());write(progress,state)
    except BaseException:
        state.update(status='failed',finished_at=now(),error=traceback.format_exc())
        write(progress,state);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['forest','robometer','robobrain25'])
    p.add_argument('--rank',type=int,required=True);a=p.parse_args();main(a.kind,a.rank)
