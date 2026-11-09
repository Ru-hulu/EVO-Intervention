
"""One independent GPU shard; preserve every augmented observation and native GRM response."""
import argparse, gc, json, math, os, sys, time, traceback
from pathlib import Path
import torch
from transformers import set_seed
from grm import Dataset, Collator, load_model, load_processor, generate, parse_score
from eval_common import write, now

ROOT=Path(__file__).resolve().parent
def main(rank):
    run=json.loads((ROOT/'run.json').read_text())
    config=json.loads((ROOT/'model-config.json').read_text())
    output=Path(run['output_dir']); path=output/f'predictions-{rank}.jsonl'
    progress=ROOT/f'worker-{rank}.json'
    state=dict(status='loading',rank=rank,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],
               started_at=now(),completed_samples=0,invalid_generations=0,batch_size=run['batch_size'])
    write(progress,state)
    try:
        set_seed(42)
        dataset=Dataset(config['data_dir'],'augmented_validation')
        wanted=set(run['shards'][rank]['episode_ids'])
        indices=[i for i,r in enumerate(dataset.rows) if r['episode_id'] in wanted]
        assert len(indices)==run['shards'][rank]['samples']
        processor=load_processor(config['model_path'],config['min_pixels'],config['max_pixels'])
        model=load_model(config['model_path'],adapter=run['checkpoint']).to('cuda').eval()
        assert not any(p.requires_grad for p in model.parameters())
        collator=Collator(processor,config['max_length'])
        pos=0;size=run['batch_size'];start=time.monotonic();last_status=start
        state.update(status='running',total_samples=len(indices),loaded_at=now())
        write(progress,state)
        with path.open('x') as f:
            while pos<len(indices):
                rows=[dataset[i] for i in indices[pos:pos+size]]
                batch=None
                try:
                    batch={k:v.to('cuda') for k,v in collator.encode(rows,supervised=False).items()}
                    assert 'labels' not in batch and 'target_pt' not in batch
                    answers=generate(model,processor,batch,config['max_new_tokens'])
                except torch.cuda.OutOfMemoryError:
                    batch=None;gc.collect();torch.cuda.empty_cache()
                    if size==1:raise
                    size=max(1,size//2)
                    state.update(batch_size=size,last_oom_at=now())
                    write(progress,state)
                    print(json.dumps(dict(event='oom_batch_reduced',rank=rank,batch_size=size)),flush=True)
                    continue
                for row,answer in zip(rows,answers,strict=True):
                    eid=row['episode_id']
                    record={k:row[k] for k in ['episode_id','sample_index','source_sample_index',
                            'anchor_sample_index','frame_index','source_frame_index','timestamp_seconds','kind']}
                    record.update(suite=eid.split('/')[0],task='/'.join(eid.split('/')[:2]),
                                  target_pt=row['pt'],raw_response=answer)
                    try:record['predicted_pt']=parse_score(answer)
                    except ValueError as exc:
                        record.update(predicted_pt=None,error=str(exc));state['invalid_generations']+=1
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                f.flush()
                pos+=len(rows);elapsed=time.monotonic()-start
                state.update(completed_samples=pos,elapsed_seconds=elapsed,samples_per_second=pos/elapsed,
                             batch_size=size,updated_at=now(),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                if pos==len(rows) or time.monotonic()-last_status>=20 or pos==len(indices):
                    write(progress,state);last_status=time.monotonic()
                if pos==len(rows):
                    print(json.dumps(dict(event='first_batch_completed',rank=rank,samples=pos,
                        invalid_generations=state['invalid_generations'],peak_allocated_gib=state['peak_allocated_gib'])),flush=True)
                del batch,answers
        state.update(status='completed' if state['invalid_generations']==0 else 'invalid_generations',finished_at=now())
        write(progress,state)
        if state['invalid_generations']:raise RuntimeError('Invalid GRM generations; predictions preserved, full metrics withheld')
    except BaseException:
        state.update(status='failed',finished_at=now(),error=traceback.format_exc())
        write(progress,state);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True)
    main(p.parse_args().rank)
