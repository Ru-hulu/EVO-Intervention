"""Batch-one native prediction with verified final adapters and per-frame resume."""
import argparse
import fcntl
import math
import os
import time
import traceback
from functools import lru_cache
from .common import *


@lru_cache(maxsize=48)
def image(path):
    from PIL import Image
    with Image.open(path) as im:return im.convert('RGB')


def check_forest_checkpoint(ck, proof):
    assert ck['epoch']==1 and ck['next_batch']==0 and len(ck['rng_by_rank'])==2
    assert ck['optimizer']['state'] and ck['step']==proof['optimizer_steps']
    if ck['config']['lr_schedule']=='constant':
        assert ck['scheduler'] is None, 'Native constant-LR trainer has no scheduler'
    else:assert ck['scheduler']['last_epoch']==ck['step']
    protocol=ck['config']['run_protocol']
    assert protocol['train_episodes']==10 and protocol['suite']=='Recover' and protocol['fps']==5


def load(name, info):
    import torch
    if name=='forest':
        from mem_lite.runtime import load_checkpoint
        from mem_lite.train_stage3 import trainable_state
        from experiments.prm_five_groups_20261114.forest_adapter import ForestRuntime,state_digest
        p=Path(info['checkpoint']);assert sha(p)==info['checkpoint_sha256']
        ck=torch.load(p,map_location='cpu',weights_only=True)
        proof=read(info['proof'])
        check_forest_checkpoint(ck,proof)
        assert ck['config']['run_protocol']['source_manifest_sha256']==read(ROOT/'inputs.json')['train_selection_sha256']
        base=read('/cache/cvpzhr/setup/qwen35-original-download-20261113/verification.json')
        assert base['all_official_hashes_verified']
        for filename,item in base['files'].items():assert sha(Path(base['destination'])/filename)==item['sha256']
        expected=state_digest(ck['model']);del ck
        model,processor,config=load_checkpoint(p,device='cuda',event_split_method='max_depth')
        actual=state_digest(trainable_state(model));assert actual==expected
        return ForestRuntime(model,processor,config),dict(loaded_tensor_digest=actual,strict_final_state=True,
            base_revision=base['revision'],official_base_files=base['files'])
    from experiments.recover_finetune_20261114.train import load_model,robometer_api
    os.environ['RANK']='0'
    cfg=info['config']
    for path,digest in cfg['weights_sha256'].items():assert sha(path)==digest
    native=read('/cache/cvpzhr/setup/recover-six-model-caches-20261114/'+name+'-native-provenance.json')
    for path,digest in native['source_sha256'].items():assert sha(path)==digest
    model,processor,original=load_model(cfg)
    p=Path(info['checkpoint']);assert sha(p/'trainable-state.pt')==info['checkpoint_sha256']
    state=torch.load(p/'trainable-state.pt',map_location='cpu',weights_only=True)
    current={n:v for n,v in model.named_parameters() if v.requires_grad}
    assert set(current)==set(state), 'Adapter/head name mismatch'
    with torch.no_grad():
        for n,v in current.items():
            assert v.shape==state[n].shape and v.dtype==state[n].dtype
            v.copy_(state[n])
    expected=read(p/'trainable-state.json')['sha256'];assert tensor_digest(current.items())==expected
    model.to('cuda').eval().requires_grad_(False)
    # No training dropout or optimizer state is used during inference.
    audit=dict(loaded_tensor_digest=expected,strict_final_state=True,base_revision=cfg['official_revision'],
        base_weight_sha256=cfg['weights_sha256'],native_source_sha256=native['source_sha256'])
    if name=='robometer-4b':return (model,processor,robometer_api()),audit
    from experiments.prm_five_groups_20261114.official_baseline import native_module
    native_worker=native_module();shell=object.__new__(native_worker.Generative)
    shell.kind='proc' if name=='procvlm-2b' else 'grm';shell.processor=processor
    shell.template=(native_worker.ROOT/('proc_prompt.txt' if shell.kind=='proc' else 'grm_prompt.txt')).read_text()
    return (model,processor,shell,native_worker),audit


def infer(name, runtime, episode, i):
    import torch
    if name=='forest':
        result=runtime.predict([image(p) for p in episode['frames'][i]['image']],episode['frames'][i]['frame_index'],episode['key'],episode['task'])
        return result['prediction'],result
    if name=='robometer-4b':
        from experiments.recover_six_caches_20261114.common import slots
        model,processor,api=runtime
        paths=[episode['frames'][j]['image'][v] for j,v in slots(name,i,len(episode['frames']))]
        assert len(paths)==10
        inputs=api.progress_batch(processor,[(episode['task'],[image(p) for p in paths])])
        inputs={k:v.to('cuda') for k,v in inputs.items()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=api.current_logits(model,inputs);value=api.progress_values(logits).item()
        return value,dict(progress_logits=logits.float().cpu().tolist()[0])
    model,processor,shell,native=runtime
    request=shell.request((episode,i,None))
    opts={k:native.MODELS[name][k] for k in ('min_pixels','max_pixels') if k in native.MODELS[name]}
    inputs=processor(text=[request['prompt']],images=request['multi_modal_data']['image'],return_tensors='pt',**opts)
    assert inputs['input_ids'].shape[1] < 16384
    inputs={k:v.to('cuda') for k,v in inputs.items()};length=inputs['input_ids'].shape[1]
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=native.MODELS[name]['max_tokens'],do_sample=False,
            use_cache=True,temperature=None,top_p=None,top_k=None)
    raw=processor.tokenizer.decode(output[0,length:],skip_special_tokens=True)
    kind=shell.kind
    value=native.parse(raw,'progress' if kind=='proc' else 'score',0 if kind=='proc' else -100,100)
    return value,raw


def validate_prefix(result, row, inputs_sha, checkpoint_sha):
    n=len(result['predictions'])
    assert result['inputs_sha256']==inputs_sha and result['checkpoint_sha256']==checkpoint_sha
    assert result['key']==row['key'] and 0<=n<=len(row['frame_indices'])
    assert result['frame_indices']==row['frame_indices'][:n]
    assert result['target_pt']==row['target_pt'][:n] and len(result['raw_responses'])==n
    return n


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=MODELS);args=p.parse_args();name=args.model
    jobs=ROOT/'jobs';jobs.mkdir(exist_ok=True)
    lock=(jobs/(name+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(model=name,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],status='initializing',
        started=time.time(),completed_episodes=0,completed_samples=0)
    def update(**kw):state.update(kw,updated=time.time());save(jobs/(name+'.json'),state)
    try:
        import torch
        torch.set_num_threads(2);torch.manual_seed(42);torch.cuda.set_per_process_memory_fraction(.65)
        manifest=read(ROOT/'inputs.json');digest=sha(ROOT/'inputs.json');info=manifest['model_checkpoints'][name]
        assert sha(info['proof'])==info['proof_sha256']
        update();runtime,audit=load(name,info)
        audit.update(model=name,regime='Recover-10-trajectory-fine-tuned',checkpoint=info['checkpoint'],
            checkpoint_sha256=info['checkpoint_sha256'],inputs_sha256=digest,
            proof_sha256=info['proof_sha256'],uses_future_endpoint_reference=name in ('robobrain25-4b','dopamine-3b'),
            inference_batch=1,raw_predictions_unchanged=True,torch=torch.__version__,cuda_memory_fraction=.65)
        audit_path=jobs/(name+'-initialization.json');save(audit_path,audit)
        baseline=host_sample();initial_oom=baseline['oom_kill'];last_resource=0
        for row in manifest['records']:
            path=result_path(name,row['key']);partial=path.with_suffix('.partial.json')
            image.cache_clear()
            if name not in ('forest','robometer-4b'):runtime[3].image.cache_clear()
            if path.exists():
                done=read(path);assert validate_prefix(done,row,digest,info['checkpoint_sha256'])==len(row['frame_indices'])
                update(completed_episodes=state['completed_episodes']+1,completed_samples=state['completed_samples']+len(row['frame_indices']));continue
            assert sha(row['progress_path'])==row['progress_sha256']
            episode=frame_inputs(row)
            result=dict(model=name,key=row['key'],task_id=row['task_id'],inputs_sha256=digest,
                checkpoint_sha256=info['checkpoint_sha256'],progress_sha256=row['progress_sha256'],
                source_video_sha256=row['video_hashes'],predictions=[],raw_responses=[],frame_indices=[],target_pt=[],
                initialization_audit=str(audit_path),timestamps=row['timestamps'],uses_future_endpoint_reference=audit['uses_future_endpoint_reference'])
            if partial.exists():
                result=read(partial);n=validate_prefix(result,row,digest,info['checkpoint_sha256'])
                update(completed_samples=state['completed_samples']+n)
            if name=='forest':
                runtime.reset()
                for i in range(len(result['predictions'])):
                    runtime.advance_memory([image(p) for p in episode['frames'][i]['image']],episode['frames'][i]['frame_index'],episode['key'],episode['task'])
            update(status='running',episode=row['key'],samples_in_episode=len(result['predictions']))
            while len(result['predictions'])<len(row['frame_indices']):
                if time.time()-last_resource>10:
                    r=host_sample();last_resource=time.time();save(jobs/(name+'-resources.json'),r)
                    if not safe(r,int(os.environ['CUDA_VISIBLE_DEVICES']),initial_oom):
                        update(status='paused_for_resources');return
                i=len(result['predictions']);value,raw=infer(name,runtime,episode,i)
                value=value if value is None or math.isfinite(value) else None
                result['predictions'].append(value);result['raw_responses'].append(raw)
                result['frame_indices'].append(row['frame_indices'][i]);result['target_pt'].append(row['target_pt'][i])
                save(partial,result)
                update(samples_in_episode=i+1,completed_samples=state['completed_samples']+1,
                    elapsed=time.time()-state['started'])
            from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics
            result['metrics']=trajectory_metrics(result['predictions'],row['target_pt'],row['timestamps'])
            result['finished']=time.time();save(path,result);partial.unlink()
            update(completed_episodes=state['completed_episodes']+1)
            if state['completed_episodes']==1:save(jobs/(name+'-real-episode-canary.json'),dict(status='passed',path=str(path),sha256=sha(path),
                samples=len(result['predictions']),invalid=sum(x is None for x in result['predictions'])))
        update(status='completed',finished=time.time())
    except BaseException:
        update(status='failed',error=traceback.format_exc());raise


if __name__=='__main__':main()
