"""One-shot cache preparation. Never starts training or scheduled automations."""
import fcntl
import os
import subprocess
import sys
import time
import traceback

from .common import *
from experiments.prm_five_groups_20261114.overlap_guard import resources,pause_reasons

ENVIRONMENTS={m:('/cache/cvpzhr/envs/robometer/bin/python' if m=='robometer-4b' else
                  '/cache/robot-baselines/envs/vlac/bin/python' if m=='vlac-2b' else
                  '/cache/cvpzhr/envs/forest-labels/bin/python') for m in MODELS[:-1]}


def launch(module,args,name,gpus='',python=PYTHON):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
        PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    env.pop('PYTHONPATH',None)
    with (ROOT/(name+'.log')).open('a') as log:
        p=subprocess.Popen([python,'-m','experiments.recover_six_caches_20261114.'+module,*args],
            cwd=ROOT/'code',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True)
    save(ROOT/(name+'-launch.json'),dict(pid=p.pid,module=module,args=args,gpus=gpus,created=time.time()))
    return p


def wait_idle(initial_oom):
    since=None
    while True:
        sample=resources()
        mapping={line.split(',')[1].strip():int(line.split(',')[0]) for line in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
        occupied={mapping[line.split(',')[0].strip()] for line in subprocess.check_output(
            ['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()}
        assert sample['oom_kill']==initial_oom,'New OOM; inspect, do not restart automatically'
        reasons={g:pause_reasons(sample,g,initial_oom) for g in [0,1]}
        safe=not ({0,1}&occupied) and not any(reasons.values())
        since=(since or time.time()) if safe else None
        save(ROOT/'status.json',dict(status='waiting_for_idle_GPUs',gpus=[0,1],occupied=sorted(occupied),
             reasons=reasons,pid=os.getpid(),updated=time.time(),training_started=False))
        if since and time.time()-since>=30:return
        time.sleep(5)


def verify():
    from mem_lite.datasets.m3_dataset import M3Dataset
    from .loader import load_input
    selection=read(ROOT/'resolved-selection.json');source_checks(selection)
    expected={r['key']:rows(epdir(r)/'samples.jsonl') for r in selection['records']};models={}
    for m in MODELS[:-1]:
        receipt=read(OUT/m/'cache-ready.json');r=rows(OUT/m/'train.jsonl')
        provenance=ROOT/f'{m}-native-provenance.json'
        assert sha(provenance)==receipt['native_provenance_sha256']
        for path,digest in read(provenance)['source_sha256'].items():assert sha(path)==digest
        assert receipt['train_sha256']==sha(OUT/m/'train.jsonl')
        assert len(r)==sum(map(len,expected.values())) and set(x['episode_id'] for x in r)==set(expected)
        for x in r:
            target=expected[x['episode_id']][x['sample_index']]
            assert x['pt']==target['pt'] and x['frame_index']==target['frame_index']
            assert x['slots']==[list(p) for p in slots(m,x['sample_index'],len(expected[x['episode_id']]))]
        for record in receipt['episodes_verified']:
            audit=read(Path(record['samples_file']).with_suffix('.verification.json'))
            for p,d in audit['pixel_sha256'].items():assert sha(p)==d
            entries=rows(record['samples_file'])
            for x in [entries[0],entries[len(entries)//2],entries[-1]]:
                batch,truth=load_input(x);assert batch['input_ids'].ndim==2 and batch['pixel_values'].numel()>0
                assert truth['pt'].numel()==1
        models[m]={k:receipt[k] for k in ['status','episodes','samples','pixel_bytes']}
    files=[str(OUT/'forest'/'train'/r['task_id']/r['sample_id']) for r in selection['records']]
    dataset=M3Dataset(files);assert len(dataset)==sum(map(len,expected.values()))
    for r,p in zip(selection['records'],files):
        a=rows(Path(p)/'samples.jsonl');b=expected[r['key']]
        assert [x['frame_index'] for x in a]==[x['frame_index'] for x in b]
        assert [x['pt'] for x in a]==[x['pt'] for x in b]
    save(OUT/'forest'/'cache-ready.json',dict(status='native_training_cache_ready',episodes=10,
        samples=len(dataset),sample_files=files,data_signature=dataset.signature,history_source=dataset.history_source))
    models['forest']=dict(status='native_training_cache_ready',episodes=10,samples=len(dataset))
    source_checks(selection)
    result=dict(status='completed',models=models,episodes_per_model=10,tasks=5,per_task=2,
        samples_per_model=len(dataset),source_splits_videos_annotations_original_pt_unchanged=True,
        training_started=False,generative_cache_scope='native model inputs + original PT; no invented SFT responses',
        selection_sha256=sha(ROOT/'selection.json'),resolved_selection_sha256=sha(ROOT/'resolved-selection.json'),
        verification_finished=time.time())
    save(ROOT/'verification.json',result);save(OUT/'verification.json',result);save(ROOT/'status.json',result)


def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'controller.pid').exists(),'Inspect existing controller instead of launching twice'
    (ROOT/'controller.pid').write_text(str(os.getpid()))
    save(ROOT/'status.json',dict(status='decoding',pid=os.getpid(),updated=time.time(),training_started=False))
    initial_oom=resources()['oom_kill']
    child=launch('prepare',['decode'],'decode');assert child.wait()==0,'Decode failure; preserve outputs'
    # CPU checks precede any full model cache and require no CUDA context.
    for model in MODELS[:-1]:
        child=launch('model_cache',[model,'--smoke'],model+'-smoke',python=ENVIRONMENTS[model])
        assert child.wait()==0,model+' native preprocessing smoke failed'
    wait_idle(initial_oom)
    child=launch('gpu_cache',['labels'],'labels',gpus='0');assert child.wait()==0,'Label worker failed'
    wait_idle(initial_oom)
    forest=[launch('gpu_cache',['forest','--worker',str(i)],'forest-'+str(i),gpus=str(i)) for i in [0,1]]
    active={};pending=list(MODELS[:-1]);failed=[]
    while active or pending or any(p.poll() is None for p in forest):
        sample=resources()
        if sample['oom_kill']>initial_oom:raise RuntimeError('New OOM; preserve all outputs, no blind retry')
        for model,p in list(active.items()):
            if p.poll() is not None:
                if p.returncode:failed.append(model)
                del active[model]
        memory_safe=sample['resident_and_shmem']<.60*sample['memory_limit'] and sample['memory_working_set']<.82*sample['memory_limit']
        if pending and len(active)<2 and memory_safe and sample['disk_free']>100*2**30:
            m=pending.pop(0);active[m]=launch('model_cache',[m],m,python=ENVIRONMENTS[m])
        save(ROOT/'status.json',dict(status='building_caches',pid=os.getpid(),active={m:p.pid for m,p in active.items()},
            pending=pending,failed=failed,forest=[{'pid':p.pid,'exit':p.poll()} for p in forest],
            updated=time.time(),training_started=False))
        time.sleep(5)
    assert not failed and all(p.returncode==0 for p in forest),(failed,[p.returncode for p in forest])
    verify()


if __name__=='__main__':
    try:main()
    except Exception:
        save(ROOT/'controller-error.json',dict(error=traceback.format_exc(),time=time.time()))
        save(ROOT/'status.json',dict(status='failed',pid=os.getpid(),error=traceback.format_exc(),updated=time.time()))
        raise
