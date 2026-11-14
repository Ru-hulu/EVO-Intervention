"""Official generative/VLAC baselines on the new frozen 385-episode manifest.

Native prompts, parsers, image ordering and recurrence come from the prior
reviewed zero-shot worker. Execution allocation is bounded for shared GPUs.
No PT labels enter model requests; invalid text remains raw/null.
"""
import argparse
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

from .prepare_cache import BASE,save,sha
from .eval_common import EVAL,PLAN,ensure_images,result_path,trajectory_metrics,aggregate

OLD=Path('/cache/cvpzhr/setup/nine-model-zeroshot-cleanval-voc-20261113')
KINDS={'robobrain25-4b':'grm','dopamine-3b':'grm','procvlm-2b':'proc','r2vlm-alfred-rl':'r2','vlac-2b':'vlac'}
COMPLETED={'completed','completed_with_input_blockers'}


def native_module():
    # The old worker imports its adjacent common.py by absolute name.
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('prm_prior_native_worker',OLD/'worker.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert Path(module.ROOT).resolve()==OLD.resolve()
    return module


def model_audit(model):
    spec=next(m for m in json.loads(PLAN.read_text())['models'] if m['id']==model)
    ref=spec['prior_provenance_reference']
    catalog=json.loads((OLD/'model-download-catalog.json').read_text())[model]
    assert catalog['revision']==ref['revision'] and catalog['local_dir']==ref['path']
    hashes={}
    for f in catalog['files']:
        if f['path'].endswith(('.safetensors','.bin')):
            assert 'lfs' in f
            digest=sha(Path(ref['path'])/f['path']);assert digest==f['lfs']['sha256'],f['path']
            hashes[f['path']]=digest
    assert hashes,'No official weight files audited'
    for path,digest in ref['small_file_sha256'].items():
        assert sha(Path(ref['path'])/path)==digest,path
        hashes[path]=digest
    return ref,hashes


def next_history(raw):
    match=re.search(r'<think>(.*?)</think>',raw,re.S)
    return match.group(1).strip() if match else raw


def vlac_next(previous,delta):
    # Native critic_to_value_simple(simple=True), in unit progress scale.
    return previous+(1-previous)*delta if previous is not None and delta is not None else None


def input_blockers(row,kind):
    issues=row.get('input_issues',[])
    return issues if kind=='grm' else [i for i in issues if i.get('view')==0]


def native_episode(row,root,kind):
    """Only instruction and image paths; truth remains in the scoring caller."""
    return {'task':row['task_prompt'],'frames':[{'image':[
        str(root/f'{index:08d}-{v}.png') if path and (kind=='grm' or v==0) else None
        for v,path in enumerate(row['videos'])]} for index in row['frame_indices']]}


def engine(native,cfg,memory_fraction):
    if cfg['kind']=='vlac':return native.VLAC(cfg)
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoProcessor
    from vllm import LLM,SamplingParams
    class BoundedGenerative(native.Generative):
        def __init__(self):
            self.kind=cfg['kind']
            self.processor=AutoProcessor.from_pretrained(cfg['path'],local_files_only=True)
            mm={k:cfg[k] for k in ['min_pixels','max_pixels'] if k in cfg}
            self.engine=LLM(model=cfg['path'],dtype='bfloat16',trust_remote_code=True,
                gpu_memory_utilization=memory_fraction,max_model_len=16384,max_num_seqs=1,
                max_num_batched_tokens=16384,limit_mm_per_prompt={'image':1 if self.kind=='r2' else 8},
                enable_prefix_caching=True,seed=42,mm_processor_kwargs=mm)
            self.sampling=SamplingParams(temperature=0,max_tokens=cfg['max_tokens'])
            self.template=(OLD/{'grm':'grm_prompt.txt','proc':'proc_prompt.txt','r2':'r2_prompt.txt'}[self.kind]).read_text()
            self.pool=ThreadPoolExecutor(2)
    return BoundedGenerative()


def evaluate(args):
    import torch
    name=f'{args.model}-group1-{args.shard}';jobs=EVAL/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    lock=(jobs/f'{name}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2);torch.manual_seed(42)
    if KINDS[args.model]=='vlac':torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    state={'model':args.model,'group':1,'shard':args.shard,'shards':4,'pid':os.getpid(),
           'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'started':time.time(),'completed_episodes':0,
           'completed_samples':0,'status':'verifying_weights','memory_fraction':args.memory_fraction,'batch':1}
    def update(**kw):state.update(kw,updated=time.time());save(jobs/f'{name}.json',state)
    update()
    try:
        manifest_path=EVAL/'group1-manifest.json';manifest=json.loads(manifest_path.read_text());digest=sha(manifest_path)
        assert manifest['plan_sha256']==sha(PLAN)
        kind=KINDS[args.model];selected=[e for i,e in enumerate(manifest['episodes']) if i%4==args.shard]
        pending=[];blocked=[]
        for row in selected:
            issues=input_blockers(row,kind)
            if issues:blocked.append({'key':row['key'],'issues':issues});continue
            path=result_path(args.model,row['key'])
            if path.exists():
                result=json.loads(path.read_text())
                assert result['manifest_sha256']==digest and result['frame_indices']==row['frame_indices']
                assert result['target_pt']==row['target_pt'] and len(result['predictions'])==len(row['target_pt'])
                state['completed_episodes']+=1;state['completed_samples']+=len(row['target_pt'])
            else:pending.append(row)
        update(total_episodes=len(selected),total_samples=sum(len(e['target_pt']) for e in selected),input_blockers=blocked)
        if not pending:
            update(status='completed_with_input_blockers' if blocked else 'completed');aggregate(args.model);return
        ref,hashes=model_audit(args.model);native=native_module();cfg=dict(native.MODELS[args.model],batch=1)
        assert cfg['kind']==kind and cfg['path']==ref['path']
        audit={'model':args.model,'official_revision':ref['revision'],'checkpoint':ref['path'],
            'official_file_sha256':hashes,'adapter_loaded':False,'config':cfg,
            'native_worker_sha256':sha(OLD/'worker.py'),'native_config_sha256':sha(OLD/'common.py'),
            'worker_sha256':sha(__file__),'manifest_sha256':digest,'plan_sha256':sha(PLAN),
            'memory_fraction':args.memory_fraction,'batch':1,'torch':torch.__version__,
            'uses_future_endpoint_reference':kind=='grm',
            'prompt_sha256':{p:sha(OLD/p) for p in ['grm_prompt.txt','proc_prompt.txt','r2_prompt.txt']}}
        update(status='loading_model');model=engine(native,cfg,args.memory_fraction)
        audit['native_model_loaded']=True;save(jobs/f'{name}-initialization.json',audit)
        started=time.time();new_samples=0;update(status='running')
        for row in pending:
            update(phase='decode_and_verify_images',episode=row['key'],samples_in_episode=0)
            root=ensure_images(row,main_only=kind!='grm');native.image.cache_clear()
            episode=native_episode(row,root,kind)
            pred,raw=([0.],[None]) if kind=='vlac' else ([],[])
            history=native.INITIAL_HISTORY;partial_path=jobs/f'{name}-partial.json'
            if partial_path.exists():
                prior=json.loads(partial_path.read_text())
                if prior['episode']==row['key']:
                    n=len(prior['predictions']);assert prior['manifest_sha256']==digest
                    assert prior['frame_indices']==row['frame_indices'][:n] and prior['target_pt']==row['target_pt'][:n]
                    assert len(prior['raw_responses'])==n
                    pred,raw=prior['predictions'],prior['raw_responses']
                    if kind=='r2' and raw:history=next_history(raw[-1])
            state['completed_samples']+=len(pred)
            while len(pred)<len(row['frame_indices']):
                if (EVAL/f'pause-{name}').exists():
                    update(status='paused_by_resource_guard');return
                i=len(pred);values,answers=model.infer([(episode,i,history)])
                assert len(values)==len(answers)==1 and isinstance(answers[0],str)
                value=values[0]
                if value is not None and not math.isfinite(value):value=None
                if kind=='vlac':value=vlac_next(pred[-1],value)
                if kind=='r2':history=next_history(answers[0])
                pred.append(value);raw.append(answers[0]);new_samples+=1
                save(partial_path,{'episode':row['key'],'manifest_sha256':digest,'predictions':pred,
                    'raw_responses':raw,'frame_indices':row['frame_indices'][:len(pred)],'target_pt':row['target_pt'][:len(pred)]})
                update(phase='inference',samples_in_episode=len(pred),completed_samples=state['completed_samples']+1,
                    samples_per_second=new_samples/max(time.time()-started,1),invalid_samples_in_episode=sum(v is None for v in pred))
            result={'model':args.model,'official_revision':ref['revision'],'checkpoint':ref['path'],'group':1,'condition':'baseline',
                'input_fps':5,'key':row['key'],'suite':row['suite'],'task_id':row['task_id'],
                'manifest_sha256':digest,'plan_sha256':sha(PLAN),'progress_sha256':row['progress_sha256'],
                'source_video_sha256':row['video_hashes'],'initialization_audit':str(jobs/f'{name}-initialization.json'),
                'predictions':pred,'raw_responses':raw,'target_pt':row['target_pt'],'frame_indices':row['frame_indices'],
                'timestamps':row['timestamps'],'uses_future_endpoint_reference':kind=='grm',
                'metrics':trajectory_metrics(pred,row['target_pt'],row['timestamps']),'finished':time.time()}
            save(result_path(args.model,row['key']),result);partial_path.unlink(missing_ok=True)
            update(completed_episodes=state['completed_episodes']+1)
        update(status='completed_with_input_blockers' if blocked else 'completed',finished=time.time());aggregate(args.model)
    except BaseException:
        update(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('model',choices=list(KINDS))
    parser.add_argument('--shard',type=int,required=True);parser.add_argument('--memory-fraction',type=float,default=.30)
    args=parser.parse_args();assert 0<=args.shard<4 and 0<args.memory_fraction<=.35
    evaluate(args)
