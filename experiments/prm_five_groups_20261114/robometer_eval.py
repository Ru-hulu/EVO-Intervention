"""Official Robometer on the new frozen validation manifest, resumable by episode."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from functools import lru_cache

from .eval_common import EVAL, OUT, PLAN, ensure_images, uniform_indices, result_path, trajectory_metrics, aggregate
from .prepare_cache import BASE, save, sha

MODEL='/cache/cvpzhr/models/Robometer-4B'
REVISION='beef63bc914c5c189329d49c6d712d96d632aa34'
OLD=Path('/cache/cvpzhr/setup/nine-model-zeroshot-cleanval-voc-20261113')


@lru_cache(maxsize=96)
def load_image(path):
    from PIL import Image
    if path is None:
        return Image.new('RGB',(64,64),(0,0,0))
    with Image.open(path) as image:
        return image.convert('RGB')


def model_file_audit():
    catalog=json.loads((OLD/'model-download-catalog.json').read_text())['robometer-4b']
    assert catalog['revision']==REVISION and catalog['local_dir']==MODEL
    verified={}
    for f in catalog['files']:
        if not f['path'].endswith('.safetensors'):
            continue
        digest=sha(Path(MODEL)/f['path'])
        assert digest==f['lfs']['sha256'],f['path']
        verified[f['path']]=digest
    for path,digest in catalog['small_file_sha256'].items():
        assert sha(Path(MODEL)/path)==digest,path
        verified[path]=digest
    return verified


def evaluate(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    sys.path[:0]=[str(OLD/'runtime'),'/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream']
    from robometer_training.model import load_processor,load_model,progress_batch,current_logits,progress_values
    name=f'robometer-4b-group1-{args.shard}'
    jobs=EVAL/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    lock=(jobs/f'{name}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'model':'robometer-4b','group':1,'shard':args.shard,'shards':args.shards,
            'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'started':time.time(),
            'completed_episodes':0,'completed_samples':0,'batch_size':args.batch,'memory_fraction':args.memory_fraction}
    def update(**kw):
        status.update(kw,updated=time.time());save(jobs/f'{name}.json',status)
    manifest_path=EVAL/'group1-manifest.json'
    manifest=json.loads(manifest_path.read_text());digest=sha(manifest_path)
    assert manifest['plan_sha256']==sha(PLAN)
    selected=[e for i,e in enumerate(manifest['episodes']) if i%args.shards==args.shard]
    pending=[];blocked=[]
    for e in selected:
        if e.get('input_issues'):
            blocked.append({'key':e['key'],'issues':e['input_issues']})
            continue
        path=result_path('robometer-4b',e['key'])
        if path.exists():
            r=json.loads(path.read_text());assert r['manifest_sha256']==digest
            assert r['frame_indices']==e['frame_indices'] and r['target_pt']==e['target_pt']
            assert len(r['predictions'])==len(e['target_pt'])
            status['completed_episodes']+=1;status['completed_samples']+=len(e['target_pt'])
        else:pending.append(e)
    update(status='verifying_official_weights',total_episodes=len(selected),total_samples=sum(len(e['target_pt']) for e in selected),input_blockers=blocked)
    try:
        hashes=model_file_audit()
        processor=load_processor(MODEL,min_pixels=4096,max_pixels=65536)
        model,_=load_model(MODEL,processor)
        assert not any('lora_' in n for n,_ in model.named_parameters())
        model.to('cuda').eval().requires_grad_(False)
        audit={'model':MODEL,'official_revision':REVISION,'official_file_sha256':hashes,
               'strict_checkpoint_load':True,'adapter_loaded':False,'progress_head':'official 10-bin softmax expectation',
               'history':'uniform prefix 8 rint, short prefix right padded current; 7 main plus current three views',
               'uses_future_frame':False,'manifest_sha256':digest,'torch':torch.__version__,
               'runtime_sha256':sha(OLD/'runtime/robometer_training/model.py'),
               'worker_sha256':sha(__file__),'plan_sha256':sha(PLAN),'memory_fraction':args.memory_fraction}
        save(jobs/f'{name}-initialization.json',audit)
        def infer(examples):
            inputs=progress_batch(processor,examples)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=current_logits(model,{k:v.to('cuda') for k,v in inputs.items()})
                values=progress_values(logits).cpu().tolist()
            if not all(__import__('math').isfinite(v) and 0<=v<=1 for v in values):
                raise ValueError('Native progress head produced invalid values')
            return values
        update(status='running')
        completed_now=0;start=time.time();batch=args.batch
        for e in pending:
            update(phase='decode_and_verify_images',episode=e['key'],samples_in_episode=0)
            root=ensure_images(e)
            load_image.cache_clear()
            def images(i):
                indices=e['frame_indices']
                paths=[root/f'{indices[j]:08d}-0.png' for j in uniform_indices(i)[:7]]
                paths += [root/f'{indices[i]:08d}-{v}.png' if path else None for v,path in enumerate(e['videos'])]
                return [load_image(str(p) if p else None) for p in paths]
            pred=[];begin=time.time()
            partial_path=jobs/f'{name}-partial.json'
            if partial_path.exists():
                prior=json.loads(partial_path.read_text())
                if prior['episode']==e['key']:
                    assert prior['manifest_sha256']==digest
                    n=len(prior['predictions'])
                    assert prior['frame_indices']==e['frame_indices'][:n] and prior['target_pt']==e['target_pt'][:n]
                    pred=list(prior['predictions'])
                    update(completed_samples=status['completed_samples']+n,resumed_prefix_samples=n)
            while len(pred)<len(e['frame_indices']):
                if (EVAL/f'pause-{name}').exists():
                    # Keep the already computed prefix for a later resume; exit without touching training.
                    update(status='paused_by_resource_guard',phase='paused')
                    return
                i=len(pred);stop=min(i+batch,len(e['frame_indices']))
                examples=[(e['task_prompt'],images(j)) for j in range(i,stop)]
                try:
                    values=infer(examples)
                except torch.cuda.OutOfMemoryError:
                    if batch==1:raise
                    del examples
                    batch=max(1,batch//2);gc.collect();torch.cuda.empty_cache()
                    update(batch_size=batch,oom_batch_reduction=True);continue
                # Before retaining any model result, check batching against native singleton input.
                if not audit.get('native_batch_consistency'):
                    reference=infer(examples[:1])[0]
                    assert abs(reference-values[0])<.002,(reference,values[0])
                    audit['native_batch_consistency']={'singleton':reference,'batched':values[0],
                                                       'absolute_difference':abs(reference-values[0])}
                    save(jobs/f'{name}-initialization.json',audit)
                pred.extend(values);completed_now+=len(values)
                partial={'episode':e['key'],'manifest_sha256':digest,'predictions':pred,
                         'frame_indices':e['frame_indices'][:len(pred)],'target_pt':e['target_pt'][:len(pred)]}
                save(jobs/f'{name}-partial.json',partial)
                update(phase='inference',samples_in_episode=len(pred),batch_size=batch,
                       completed_samples=status['completed_samples']+len(values),
                       samples_per_second=completed_now/max(time.time()-start,1),
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            result={'model':'robometer-4b','official_revision':REVISION,'checkpoint':MODEL,'group':1,
                    'condition':'baseline','input_fps':5,'key':e['key'],'suite':e['suite'],'task_id':e['task_id'],
                    'manifest_sha256':digest,'plan_sha256':sha(PLAN),'progress_sha256':e['progress_sha256'],
                    'source_video_sha256':e['video_hashes'],'initialization_audit':str(jobs/f'{name}-initialization.json'),
                    'predictions':pred,'target_pt':e['target_pt'],'frame_indices':e['frame_indices'],
                    'timestamps':e['timestamps'],'raw_responses':[None]*len(pred),'uses_future_endpoint_reference':False,
                    'metrics':trajectory_metrics(pred,e['target_pt'],e['timestamps']),
                    'finished':time.time(),'inference_seconds':time.time()-begin}
            save(result_path('robometer-4b',e['key']),result)
            (jobs/f'{name}-partial.json').unlink(missing_ok=True)
            update(completed_episodes=status['completed_episodes']+1)
        update(status='completed_with_input_blockers' if blocked else 'completed',phase='done',finished=time.time())
        aggregate('robometer-4b')
    except BaseException:
        update(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--memory-fraction',type=float,default=.22)
    args=p.parse_args()
    assert 0<=args.shard<args.shards and args.batch>=1 and 0<args.memory_fraction<=.3
    evaluate(args)
