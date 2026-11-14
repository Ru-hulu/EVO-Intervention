"""One explicitly authorized O18 validation illustration, outside group-1 totals.

Reuses frozen PT and official inference implementations. Only this task's child
processes may be paused; never signals training or the baseline dispatcher.
"""
import argparse
import csv
import fcntl
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

from experiments.prm_five_groups_20261114.prepare_cache import save, sha, BASE
from experiments.prm_five_groups_20261114.eval_common import PLAN, EVAL, sampling_indices, ensure_images, trajectory_metrics
from experiments.prm_five_groups_20261114.eval_dispatch import gpu_processes, gpu_minors
from experiments.prm_five_groups_20261114.eval_handoff import verify_command, descendants
from experiments.prm_five_groups_20261114.evaluation_priority import matches
from experiments.prm_five_groups_20261114.overlap_guard import resources, pause_reasons

SETUP=Path('/cache/cvpzhr/setup/o18-single-trajectory-20261114')
OUT=Path('/cache/cvpzhr/evaluations/o18-single-trajectory-20261114')
DATA=Path('/cache/cvpzhr/datasets/ProPriMem/Order')
CODE=Path(__file__).resolve().parents[2]
PYTHON='/cache/cvpzhr/envs/forest-labels/bin/python'
MODELS=['robometer-4b','vlac-2b','robobrain25-4b','dopamine-3b','procvlm-2b','r2vlm-alfred-rl']
LABEL_PROTOCOL=Path('/cache/cvpzhr/setup/proprimem-visual-pt-20261113/protocol.json')
VISION_SHA='1cc7c705e96c07fb52bcb1fb32f5e05271cc249fbc619c1128fd4eddbea47aea'
GPU=2


def read(path):return json.loads(Path(path).read_text())


def select_source(manifest, metadata):
    task,=[t for t in manifest['tasks'] if t['task_id']=='O18']
    sample=task['val'][0]
    assert sample=='evomind_O18_episode_000028' and sample not in task['train']
    row,=[r for r in metadata if r['sample_id']==sample]
    assert row['task_id']=='O18' and row['total_frames']==1724 and row['fps']==30
    return task,row


def prepare():
    import av
    SETUP.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(DATA/'manifest.json');task=next(t for t in manifest['tasks'] if t['task_id']=='O18')
    folder=DATA/task['task_path'];meta=[json.loads(l) for l in (folder/'metadata.jsonl').read_text().splitlines() if l]
    task,source=select_source(manifest,meta)
    videos=[];hashes=[]
    for v in ['main_view','left_view','right_view']:
        info=source['video_views'][v];p=folder/info['video_path'];assert sha(p)==info['video_sha256']
        with av.open(str(p)) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            assert Fraction(str(stream.average_rate))==30;assert sum(1 for _ in container.decode(stream))==source['total_frames']
        videos.append(str(p));hashes.append(info['video_sha256'])
    row=dict(key='Order/O18/'+source['sample_id'],suite='Order',task_id='O18',split='val',
             sample_id=source['sample_id'],frames=source['total_frames'],fps=30,source_rate='30',
             videos=videos,video_hashes=hashes,task_prompt=source['task_prompt'],input_fps=5,input_issues=[])
    row['frame_indices']=sampling_indices(row['frames'],'30');row['timestamps']=[i/30 for i in row['frame_indices']]
    assert len(row['frame_indices'])==289
    value=dict(schema='o18-independent-illustration-v1',user_authorized=True,selection='first listed validation episode, without inspecting predictions',
        dataset_version=manifest['dataset_version'],dataset_manifest_sha256=sha(DATA/'manifest.json'),
        metadata_sha256=sha(folder/'metadata.jsonl'),row=row,models=MODELS,
        inherited_plan_sha256=sha(PLAN),original_group1_manifest_sha256=sha(EVAL/'group1-manifest.json'),
        state_training_config_sha256=sha(BASE/'State/training.config.json'),
        label_protocol=read(LABEL_PROTOCOL),label_protocol_sha256=sha(LABEL_PROTOCOL),
        include_in_group1_results=False,source_dataset_files_modified=False)
    path=SETUP/'source.json'
    if path.exists():assert read(path)==value,'Frozen illustration source changed'
    else:save(path,value)
    ensure_images(row);ensure_images(row,main_only=True)
    save(SETUP/'preparation.json',dict(status='passed',source_sha256=sha(path),observations=289,source_frames=1724))
    return value


def label():
    import torch
    import numpy as np
    from mem_lite.memory.vision import load_vision_encoder
    from mem_lite.datasets.prepare_progress import extract_deltas
    from mem_lite.datasets.prepare_proprimem import whole_video_progress, write_progress
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(.15)
    source=read(SETUP/'source.json');row=source['row'];protocol=source['label_protocol']
    assert protocol['batch_size']==16 and protocol['max_pixels']==262144
    visual,processor,digest=load_vision_encoder(protocol['model_path'],'cuda')
    assert digest==VISION_SHA and not visual.training and not any(p.requires_grad for p in visual.parameters())
    save(SETUP/'label-initialization.json',dict(vision_sha256=digest,model_path=protocol['model_path'],
         frozen=True,dtype=str(next(visual.parameters()).dtype),all_original_frames=True,
         annotation_boundaries_used=False,label_protocol_sha256=source['label_protocol_sha256']))
    started=time.time();deltas,info=extract_deltas(row['videos'][0],visual,processor,batch_size=16,max_pixels=262144)
    assert len(deltas)==1724 and info['fps']==30 and sha(row['videos'][0])==row['video_hashes'][0]
    target=whole_video_progress(deltas)
    assert target[0]==0 and target[-1]==1 and np.isfinite(target).all() and np.all(np.diff(target)>=0)
    dest=OUT/'labels/episode_000028.progress.csv';dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists(),'Preserve pre-existing labels; inspect before rerun'
    write_progress(dest,range(len(target)),deltas,target)
    row=dict(row,progress_path=str(dest),progress_sha256=sha(dest),target_pt=[float(target[i]) for i in row['frame_indices']])
    save(SETUP/'manifest.json',dict(schema='o18-single-trajectory-5fps-v1',source_sha256=sha(SETUP/'source.json'),row=row))
    save(SETUP/'label-verification.json',dict(status='passed',all_frames=1724,observations=289,
        csv_sha256=sha(dest),vision_sha256=digest,first_pt=float(target[0]),last_pt=float(target[-1]),
        delta_sum=float(deltas.sum()),seconds=time.time()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


def entry_for(row):
    from experiments.prm_five_groups_20261114.stream_manifests import digest
    value={k:row[k] for k in ['key','suite','task_id','progress_sha256','input_issues','source_rate','target_pt']}
    value.update(source_frame_indices=row['frame_indices'],playback_timestamps=row['timestamps'],
                 memory_enabled=True,all_views_share_source_frame_map=True,input_fps=5,condition='baseline')
    value['map_sha256']=digest(value)
    return value


def worker(model):
    from experiments.prm_five_groups_20261114.experiment_backends import load_backend
    assert model in MODELS
    jobs=SETUP/'jobs';jobs.mkdir(exist_ok=True);output=OUT/'predictions'/f'{model}.json'
    lock=(jobs/f'{model}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=read(SETUP/'manifest.json');row=manifest['row'];entry=entry_for(row);digest=sha(SETUP/'manifest.json')
    partial=jobs/f'{model}-partial.json';pred=[];raw=[]
    if partial.exists():
        prior=read(partial);assert prior['manifest_sha256']==digest
        pred,raw=prior['predictions'],prior['raw_responses'];assert len(pred)==len(raw)<=289
    state=dict(model=model,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],expected_observations=289,started=time.time())
    def update(**kw):state.update(kw,updated=time.time());save(jobs/f'{model}.json',state)
    if output.exists():
        r=read(output);assert r['manifest_sha256']==digest and len(r['predictions'])==289
        update(status='completed',completed_observations=289);return
    update(status='loading',completed_observations=len(pred))
    try:
        backend=load_backend(model,'Order',True,.22 if model=='robometer-4b' else .30)
        save(jobs/f'{model}-initialization.json',dict(backend.audit,manifest_sha256=digest,independent_illustration=True))
        root=ensure_images(row,main_only=model not in ['robometer-4b','robobrain25-4b','dopamine-3b'])
        backend.begin(row,entry,root,{'predictions':pred,'raw_responses':raw},lambda:(SETUP/'pause').exists())
        while len(pred)<289:
            if (SETUP/'pause').exists():update(status='paused_by_resource_guard',completed_observations=len(pred));return
            value,answer,_=backend.predict(len(pred),pred,raw)
            if value is not None:assert math.isfinite(value)
            pred.append(value);raw.append(answer)
            save(partial,dict(manifest_sha256=digest,predictions=pred,raw_responses=raw))
            update(status='running',completed_observations=len(pred),invalid_predictions=sum(v is None for v in pred))
        save(output,dict(schema='o18-qualitative-prediction-v1',model=model,key=row['key'],
            manifest_sha256=digest,initialization_sha256=sha(jobs/f'{model}-initialization.json'),
            checkpoint=backend.audit['checkpoint'],official_revision=backend.audit['official_revision'],
            predictions=pred,raw_responses=raw,frame_indices=row['frame_indices'],timestamps=row['timestamps'],
            target_pt=row['target_pt'],progress_sha256=row['progress_sha256'],source_video_sha256=row['video_hashes'],
            uses_future_endpoint_reference=backend.audit['uses_future_endpoint_reference'],
            metrics=trajectory_metrics(pred,row['target_pt'],row['timestamps']),include_in_group1_results=False,finished=time.time()))
        partial.unlink(missing_ok=True);update(status='completed',completed_observations=289)
    except BaseException:update(status='failed',error=traceback.format_exc());raise


def baseline_evidence():
    now=time.time();dispatch=read(EVAL/'dispatcher-status.json');assert now-dispatch['updated']<90
    receipt=dispatch['active'][str(GPU)];assert receipt['group']==1 and receipt['model']=='r2vlm-alfred-rl'
    ident=verify_command(receipt);tree=descendants(receipt['pid']);minor=gpu_minors()[GPU];mapped=[]
    for node in tree:
        p=Path('/proc')/str(node['pid']);maps=(p/'maps').read_text()
        devs={int(m.group(1)) for line in maps.splitlines() if (m:=re.search(r'/dev/nvidia(\d+)$',line))}
        if devs:assert devs=={minor};mapped.append(node)
    assert mapped,'No matching CUDA descendant'
    contexts=gpu_processes()[GPU];assert len(contexts)==1,'Need exactly the recorded baseline context'
    job=read(EVAL/'jobs'/f"{receipt['name']}.json");assert job['status']=='running' and now-job['updated']<60
    return dict(receipt=receipt,identity=ident,tree=tree,mapped=mapped,host_pid=contexts[0],
        completed_samples=job['completed_samples'],observed_at=now)


def controller():
    SETUP.mkdir(parents=True,exist_ok=True)
    lock=(SETUP/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (SETUP/'controller.pid').write_text(str(os.getpid()))
    assert read(SETUP/'preparation.json')['status']=='passed'
    initial_oom=resources()['oom_kill'];initial=baseline_evidence();start=time.time()
    def update(phase,**kw):save(SETUP/'status.json',dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw))
    update('validating_60_seconds_single_context_ownership')
    while time.time()-start<60:
        evidence=baseline_evidence();sample=resources()
        assert evidence['identity']==initial['identity'] and evidence['host_pid']==initial['host_pid']
        assert not pause_reasons(sample,GPU,initial_oom)
        assert sample['gpus'][GPU]['total_mib']-sample['gpus'][GPU]['used_mib']>=40960
        time.sleep(5)
    evidence=baseline_evidence();assert evidence['completed_samples']>initial['completed_samples']
    save(SETUP/'gpu-admission.json',dict(status='passed',initial=initial,final=evidence,
        extra_jobs_at_once=1,maximum_compute_contexts_on_gpu=2,existing_workers_signaled=False))
    tasks=['label',*MODELS]
    for name in tasks:
        dest=SETUP/'label-verification.json' if name=='label' else OUT/'predictions'/f'{name}.json'
        if dest.exists():continue
        current=baseline_evidence();assert current['identity']==initial['identity'] and current['host_pid']==initial['host_pid']
        sample=resources();assert not pause_reasons(sample,GPU,initial_oom)
        assert sample['gpus'][GPU]['total_mib']-sample['gpus'][GPU]['used_mib']>=40960
        (SETUP/'pause').unlink(missing_ok=True)
        python=(PYTHON if name=='label' else '/cache/cvpzhr/envs/robometer/bin/python' if name=='robometer-4b'
                else '/cache/robot-baselines/envs/vlac/bin/python' if name=='vlac-2b' else '/cache/robot-baselines/envs/procvlm/bin/python')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPU),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
            HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1',
            PYTHONUNBUFFERED='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTORCH_ALLOC_CONF='expandable_segments:True')
        env.pop('PYTHONPATH',None)
        command=[python,'-m','experiments.o18_single_trajectory_20261114.run', 'label' if name=='label' else 'worker']
        if name!='label':command+=['--model',name]
        with (SETUP/f'{name}.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        update('running',job=name,child_pid=proc.pid);seen={};unsafe_since=None
        while proc.poll() is None:
            for node in descendants(proc.pid):seen[(node['pid'],node['start_ticks'])]=node
            sample=resources();contexts=gpu_processes()[GPU];reasons=pause_reasons(sample,GPU,initial_oom)
            if not matches(initial['identity']) or initial['host_pid'] not in contexts:reasons.append('baseline_identity_changed')
            if len(contexts)>2:reasons.append('unexpected_extra_GPU_context')
            if reasons:
                (SETUP/'pause').touch();unsafe_since=unsafe_since or time.time()
                update('pausing_own_job_only',job=name,reasons=reasons)
                if name=='label' or time.time()-unsafe_since>30:os.killpg(proc.pid,signal.SIGTERM)
            time.sleep(3)
        deadline=time.time()+45
        while any(matches(r) for r in seen.values()) and time.time()<deadline:time.sleep(1)
        assert not any(matches(r) for r in seen.values()),'Own GPU children remain; no next launch'
        assert proc.returncode==0 and dest.exists(),f'{name} stopped; inspect saved log and partials, no blind retry'
        assert matches(initial['identity']) and gpu_processes()[GPU]==[initial['host_pid']]
        update('job_completed',job=name)
    source=read(SETUP/'source.json');assert sha(EVAL/'group1-manifest.json')==source['original_group1_manifest_sha256']
    assert sha(BASE/'State/training.config.json')==source['state_training_config_sha256']
    results={m:read(OUT/'predictions'/f'{m}.json') for m in MODELS}
    assert all(len(r['predictions'])==289 for r in results.values())
    save(SETUP/'completion.json',dict(status='passed',models=MODELS,observations_per_model=289,
         baseline_manifest_unchanged=True,training_config_unchanged=True,existing_workers_signaled=False,
         invalid_predictions={m:sum(v is None for v in r['predictions']) for m,r in results.items()},finished=time.time()))
    update('completed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','label','worker','run']);p.add_argument('--model',choices=MODELS);a=p.parse_args()
    try:
        if a.mode=='prepare':prepare()
        elif a.mode=='label':label()
        elif a.mode=='worker':worker(a.model)
        else:controller()
    except BaseException:
        save(SETUP/f'{a.mode}-error.json',dict(error=traceback.format_exc(),at=time.time()));raise
