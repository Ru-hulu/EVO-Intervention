"""Dedicated Count cache preparation followed by fresh two-GPU one-epoch training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys
import time
import traceback

from experiments.prm_five_groups_20261114.prepare_cache import sha, save
from experiments.prm_five_groups_20261114.overlap_guard import resources, pause_reasons, alive
from .prepare_run import ROOT, OUTPUT

PYTHON = '/cache/cvpzhr/envs/forest-legacy-20261114/bin/python'
CODE = ROOT / 'code'


def wait_for_idle_pair(initial_oom, phase):
    safe_since = None
    while True:
        sample = resources()
        mapping = {r.split(',')[1].strip():int(r.split(',')[0]) for r in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
        occupied = set()
        for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines():
            occupied.add(mapping[line.split(',')[0].strip()])
        reasons = {g:pause_reasons(sample,g,initial_oom) for g in (0,1)}
        healthy = not ({0,1} & occupied) and not any(reasons.values())
        safe_since = (safe_since or time.time()) if healthy else None
        if sample['oom_kill'] > initial_oom:
            raise RuntimeError('New container OOM; preserve existing work and inspect')
        if safe_since and time.time()-safe_since >= 30:
            return
        save(ROOT/'pipeline-status.json',dict(status='waiting_for_idle_safe_gpus',next_phase=phase,
             pid=os.getpid(),gpus=[0,1],occupied=sorted(occupied),reasons=reasons,updated=time.time()))
        time.sleep(5)


def main(reuse_verified_cache=False):
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if reuse_verified_cache:
        assert not OUTPUT.exists(), 'Never replace a started training run'
        assert json.loads((ROOT/'pipeline-status.json').read_text())['status']=='failed'
        assert json.loads((ROOT/'cache-verification.json').read_text())['status']=='passed'
        old_start=json.loads((ROOT/'pipeline-started.json').read_text())
        assert not alive(old_start['pid']), 'Old pipeline must have exited'
        for path in [*ROOT.glob('cache-worker-*.json'),ROOT/'training-status.json']:
            assert not alive(json.loads(path.read_text()).get('pid')), f'Old process still alive: {path}'
        history=ROOT/'recovery-history'/str(time.time_ns());history.mkdir(parents=True)
        for name in ['pipeline-started.json','pipeline-status.json','pipeline.log','training-status.json',
                     'training-runner.log','training.config.json','source-checksums.json']:
            path=ROOT/name
            if path.exists():shutil.copy2(path,history/name)
    else:
        assert not (ROOT/'pipeline-started.json').exists(), 'Inspect an existing run instead of launching twice'
    subset=json.loads((ROOT/'subset-manifest.json').read_text())
    assert subset['selected_episodes']==35 and subset['task_count']==15
    assert all(t['selected']>=2 for t in subset['tasks'].values())
    (ROOT/'pipeline.pid').write_text(str(os.getpid()))
    save(ROOT/'pipeline-started.json',dict(pid=os.getpid(),started=time.time()))
    initial_oom=resources()['oom_kill']
    if not reuse_verified_cache:wait_for_idle_pair(initial_oom,'caching')
    children=[]
    for worker in (range(4) if not reuse_verified_cache else []):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(worker%2),OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
        env.pop('PYTHONPATH',None)
        with (ROOT/f'cache-worker-{worker}.log').open('w') as log:
            child=subprocess.Popen([PYTHON,'-m','experiments.forest_count35_20261114.cache_worker',str(worker)],
                cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
    state=dict(status='caching',started=time.time(),pid=os.getpid(),worker_pids=[p.pid for p in children],
               gpus=[0,1],suite='Count',episodes=35,tasks=15,automatic_training=True,
               reused_verified_cache=reuse_verified_cache)
    save(ROOT/'pipeline-status.json',state)
    codes=[p.wait() for p in children]
    assert all(c==0 for c in codes), f'Cache worker errors: {codes}; preserve every output'
    with sqlite3.connect(ROOT/'cache-queue.sqlite') as db:
        rows=db.execute('SELECT payload,status,samples FROM jobs ORDER BY key').fetchall()
    assert len(rows)==35 and all(r[1]=='completed' for r in rows)
    assert sha(subset['source_manifest'])==subset['source_manifest_sha256']
    assert all(sha(p)==digest for p,digest in subset['source_metadata_sha256'].items())
    files=[]
    for payload,_,samples in rows:
        r=json.loads(payload);p=Path(r['output']);v=json.loads((p/'cache-verification.json').read_text())
        assert sha(p/'samples.jsonl')==v['samples_sha256']
        assert sha(r['progress_path'])==v['progress_sha256']==r['progress_sha256']
        assert v['samples']==samples
        files.append(str(p))
    from mem_lite.datasets.m3_dataset import M3Dataset
    dataset=M3Dataset(files);assert len(dataset)==sum(r[2] for r in rows)
    config=json.loads((ROOT/'reference-training.config.json').read_text())
    assert dataset.history_source==config['history_source']
    config.update(sample_files=files,output_dir=str(OUTPUT),data_signature=dataset.signature,
                  epochs=1,max_steps=None,save_every_steps=None,save_every_epochs=1)
    config['run_protocol']=dict(suite='Count',world_size=2,gpus=[0,1],effective_batch=8,fps=5,
        initialization='fresh original Qwen3.5-4B; new LoRA/history projection/progress head',
        train_episodes=35,train_samples=len(dataset),epochs=1,artificial_augmentation=False,
        old_training_weights_reused=False,other_suites_enabled=False,subsampling_seed=42,
        subsampling_unit='complete_episode',minimum_per_task=2,tasks=15,exact_episode_count=35,
        original_train_episodes=699,source_manifest_sha256=subset['source_manifest_sha256'],fla_version='0.3.2')
    save(ROOT/'training.config.json',config)
    save(ROOT/'cache-verification.json',dict(status='passed',episodes=35,samples=len(dataset),
        task_count=15,minimum_per_task=2,source_splits_unchanged=True,source_labels_unchanged=True,
        all_sample_hashes_verified=True,history_source=dataset.history_source))
    wait_for_idle_pair(initial_oom,'training')
    state.update(status='training',cache_finished=time.time(),train_samples=len(dataset))
    save(ROOT/'pipeline-status.json',state)
    with (ROOT/'training-runner.log').open('w') as log:
        result=subprocess.run([PYTHON,'-m','experiments.forest_count35_20261114.run_training'],cwd=CODE,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0, f'Training runner failed: {result.returncode}'
    final=json.loads((ROOT/'training-status.json').read_text())
    assert final['status']=='completed' and final['final_checkpoint_verified']
    state.update(status='completed',finished=time.time(),checkpoint=final['checkpoint'])
    save(ROOT/'pipeline-status.json',state)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-verified-cache',action='store_true')
    args=parser.parse_args()
    try:
        main(args.reuse_verified_cache)
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        save(ROOT/'pipeline-status.json',dict(status='failed',error=error,pid=os.getpid(),finished=time.time()))
        raise
