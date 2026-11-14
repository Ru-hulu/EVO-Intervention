"""GPU2/3 only: bounded throughput trials, then fresh one-epoch Recover FoREST."""
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path('/cache/cvpzhr/setup/forest-recover-10-legacy-e1-20261114')
OUTPUT = Path('/cache/cvpzhr/outputs/forest-recover-10-legacy-2gpu-e1-20261114')
CACHE = Path('/cache/cvpzhr/datasets/recover-six-model-caches-5fps-20261114/forest')
CACHE_SETUP = Path('/cache/cvpzhr/setup/recover-six-model-caches-20261114')
REFERENCE = Path('/cache/cvpzhr/setup/forest-count-35-legacy-e1-20261114')
PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
GPUS = (2, 3)


def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def save(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def idle():
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'],text=True).splitlines():
        index,used=map(int,line.split(','))
        if index in GPUS: assert used<1000,f'GPU{index} occupied; refusing to share another training task'


def prepare():
    assert not OUTPUT.exists(), 'Preserve any prior formal output'
    assert read(CACHE_SETUP/'final-verification.json')['status']=='passed'
    receipt=read(CACHE/'cache-ready.json')
    assert receipt['episodes']==10 and receipt['samples']==2550
    config=read(REFERENCE/'training.config.json')
    assert config['epochs']==1 and config['model']['gradient_checkpointing'] is False
    assert config['lr_schedule']=='constant' and config['history_source']==receipt['history_source']
    config.update(sample_files=receipt['sample_files'],output_dir=str(OUTPUT),
                  data_signature=receipt['data_signature'],log_every_steps=10,
                  save_every_steps=None,save_every_epochs=1,cuda_memory_fraction=.95)
    config['run_protocol'].update(suite='Recover',gpus=list(GPUS),train_episodes=10,train_samples=2550,
        tasks=5,exact_episode_count=10,original_train_episodes=130,minimum_per_task=2,
        source_manifest_sha256=sha(CACHE_SETUP/'selection.json'),artificial_augmentation=False)
    signature=hashlib.sha256();samples=0;sources={};tasks={}
    for path in config['sample_files']:
        path=Path(path);summary=read(path/'summary.json');raw=(path/'samples.jsonl').read_bytes()
        signature.update(raw);signature.update(json.dumps(summary,sort_keys=True).encode())
        observed=[json.loads(x) for x in raw.splitlines() if x.strip()]
        assert len(observed)==summary['num_samples']
        samples+=len(observed)
        task=path.parent.name;tasks[task]=tasks.get(task,0)+1
        sources[str(path/'samples.jsonl')]=sha(path/'samples.jsonl')
        sources[str(path/'summary.json')]=sha(path/'summary.json')
    assert samples==2550 and len(tasks)==5 and set(tasks.values())=={2}
    assert signature.hexdigest()==receipt['data_signature']
    save(ROOT/'source-audit.json',dict(status='passed',episodes=10,samples=2550,tasks=tasks,
        source_hashes=sources,cache_audit_sha256=sha(CACHE_SETUP/'final-verification.json'),
        data_signature=receipt['data_signature'],other_gpus_untouched=True))
    save(ROOT/'authorization.json',dict(user_instruction='GPU2/3训练Forest，其他四模型由工作区1使用GPU0/1',
        epochs=1,gpus=list(GPUS),official_base=True,augmentation=False,other_models_scheduled=False,
        desktop_automations_created=False))
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    assert importlib.metadata.version('flash-linear-attention')=='0.3.2'
    assert callable(qwen.chunk_gated_delta_rule) and callable(qwen.FusedRMSNormGated)
    assert qwen.causal_conv1d_fn is None
    save(ROOT/'environment.json',dict(python=PYTHON,packages={n:importlib.metadata.version(n)
        for n in ('torch','transformers','triton','flash-linear-attention','fla-core','peft')},
        accelerated_linear_attention=True,gradient_checkpointing=False))
    return config


def launch(config, label, trial=False):
    idle()
    config_path=ROOT/(label+'.config.json');save(config_path,config)
    assert not Path(config['output_dir']).exists(),config['output_dir']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
        TRITON_CACHE_DIR=str(ROOT/'triton-cache'))
    env.pop('FOREST_NATIVE_GATED_NORM',None);env.pop('PRM_CUDA_MEMORY_FRACTION',None)
    command=[PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             '-m','mem_lite.train_stage3','--config',str(config_path)]
    with (ROOT/(label+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT/'code',env=env,stdout=log,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    status=dict(status='benchmarking' if trial else 'training',label=label,pid=process.pid,
        runner_pid=os.getpid(),started=time.time(),gpus=list(GPUS),batch_per_gpu=config['batch_size'],
        effective_batch=2*config['batch_size'],epochs=1,command=command,output=config['output_dir'])
    save(ROOT/'training-status.json',status)
    rc=process.wait();status.update(returncode=rc,finished=time.time())
    save(ROOT/(label+'.status.json'),status)
    return rc,status


def benchmark(config,batch):
    candidate=copy.deepcopy(config)
    label=f'benchmark-b{batch}'
    candidate.update(batch_size=batch,max_steps=14,log_every_steps=1,save_every_epochs=None,
                     output_dir=str(ROOT/'benchmarks'/label))
    candidate['run_protocol']['effective_batch']=2*batch
    rc,status=launch(candidate,label,True)
    if rc:
        detail=(ROOT/(label+'.log')).read_text()
        if not any(t in detail.lower() for t in ('out of memory','cuda error: out of memory')):
            raise RuntimeError(f'Benchmark {label} failed for a reason other than memory; inspect log')
        return dict(batch=batch,status='out_of_memory',returncode=rc)
    metrics=rows(Path(candidate['output_dir'])/'train.jsonl')
    assert len(metrics)==14 and all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in metrics)
    steady=[r for r in metrics if r['step']>=5]
    seconds=[r['step_seconds'] for r in steady]
    result=dict(batch=batch,status='passed',median_samples_per_second=statistics.median(
        r['samples_per_second'] for r in steady),mean_samples_per_second=sum(r['samples'] for r in steady)/sum(seconds),
        median_step_seconds=statistics.median(seconds),peak_allocated_gib=max(max(r['peak_allocated_gib_per_gpu']) for r in metrics),
        peak_reserved_gib=max(max(r['peak_reserved_gib_per_gpu']) for r in metrics),
        steps=14,warmup_steps_excluded=4,config=str(ROOT/(label+'.config.json')),status_file=str(ROOT/(label+'.status.json')))
    # Trial checkpoints are deliberately not reused for the one-epoch experiment.
    for path in Path(candidate['output_dir']).glob('*.pt'):path.unlink()
    return result


def main():
    lock=(ROOT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'runner.pid').write_text(str(os.getpid())+'\n')
    assert not (ROOT/'training.config.json').exists(),'Do not launch this run twice'
    config=prepare();results=[]
    for batch in (4,6):
        results.append(benchmark(config,batch));save(ROOT/'benchmark-results.json',results)
    last=results[-1]
    if last['status']=='passed' and last['peak_allocated_gib']<65:
        results.append(benchmark(config,7));save(ROOT/'benchmark-results.json',results)
    eligible=[r for r in results if r['status']=='passed' and r['peak_reserved_gib']<74]
    assert eligible,'No safe stable batch size found'
    best=max(eligible,key=lambda r:r['mean_samples_per_second'])
    config['batch_size']=best['batch'];config['run_protocol']['effective_batch']=2*best['batch']
    save(ROOT/'chosen-batch.json',dict(chosen=best,trials=results,criterion='highest measured mean samples/s below74GiB reserved',
        expected_steps=math.ceil(math.ceil(2550/2)/best['batch']),
        approximate_compute_seconds=2550/best['mean_samples_per_second']))
    rc,status=launch(config,'training',False)
    if rc:
        status['status']='failed';save(ROOT/'training-status.json',status);return rc
    import torch
    path=OUTPUT/'epoch-01.pt';checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    expected=math.ceil(math.ceil(2550/2)/best['batch'])
    assert checkpoint['epoch']==1 and checkpoint['next_batch']==0 and checkpoint['step']==expected
    assert len(checkpoint['rng_by_rank'])==2 and checkpoint['optimizer']['state'] and checkpoint['scheduler'] is None
    assert checkpoint['config']['data_signature']==config['data_signature']
    assert all(torch.isfinite(value).all() for group in checkpoint['model'].values() for value in group.values())
    audits=[read(OUTPUT/f'initialization-rank{rank}-fresh.json') for rank in (0,1)]
    assert all(a['fresh_language_lora'] and not a['resume'] for a in audits)
    assert audits[0]['initial_trainable_sha256']==audits[1]['initial_trainable_sha256']
    for source,digest in read(ROOT/'source-audit.json')['source_hashes'].items():assert sha(source)==digest
    status.update(status='completed',checkpoint=str(path),checkpoint_sha256=sha(path),
        final_checkpoint_verified=True,fresh_initialization_verified=True,optimizer_steps=expected,
        data_unchanged=True,epochs_completed=1)
    save(ROOT/'training-status.json',status)
    print(json.dumps(status),flush=True);return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        save(ROOT/'training-status.json',dict(status='failed',error=traceback.format_exc(),updated=time.time()))
        raise
