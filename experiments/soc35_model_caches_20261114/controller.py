"""One-shot CPU cache workers. No training, GPU allocation, or recurring schedule."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .build import ROOT, OUT, SUITES, MODELS, read, sha, save


def run():
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (ROOT/'controller.pid').write_text(str(os.getpid())+'\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    python = {'robometer':'/cache/cvpzhr/envs/robometer/bin/python',
              'procvlm':'/cache/robot-baselines/envs/procvlm/bin/python'}
    jobs=[]; started=time.time()
    for model in MODELS:
        assert read(ROOT/f'{model}-State-status.json')['status'] == 'smoke_passed'
    for model in MODELS:
        for suite in SUITES:
            log=(ROOT/f'{model}-{suite}.log').open('a')
            args=[python[model],'-m','experiments.soc35_model_caches_20261114.build',
                  '--model',model,'--suite',suite]
            child=subprocess.Popen(args, cwd=ROOT/'code', env=env, stdout=log, stderr=subprocess.STDOUT)
            log.close();jobs.append((model,suite,child))
    while True:
        state=[dict(model=m,suite=s,pid=p.pid,returncode=p.poll()) for m,s,p in jobs]
        pending=any(r['returncode'] is None for r in state)
        save(ROOT/'pipeline-status.json',dict(status='running' if pending else 'verifying',pid=os.getpid(),
            started=started,updated=time.time(),jobs=state,training_started=False,gpu_used=False))
        if not pending:break
        time.sleep(5)
    failures=[r for r in state if r['returncode']!=0]
    if failures:
        save(ROOT/'pipeline-status.json',dict(status='failed',jobs=state,failures=failures,
            started=started,finished=time.time(),training_started=False));return 1
    selection=read(ROOT/'selection.json')
    for ep in selection['episodes']:
        assert sha(ep['source_samples'])==ep['source_samples_sha256']
        assert sha(ep['progress_path'])==ep['progress_sha256']
        assert sha(ep['observations'])==ep['observations_sha256']
    reports=[]
    for model in MODELS:
        for suite in SUITES:
            report=read(OUT/model/suite/'cache-ready.json')
            assert report['status']=='completed' and report['episodes']==35
            assert sha(report['train_jsonl'])==report['train_sha256']
            reports.append({k:v for k,v in report.items() if k!='receipts'})
    total=sum(r['observations'] for r in reports)
    assert total==2*19479
    save(OUT/'cache-ready.json',dict(status='completed',distinct_episodes=105,
        observations_per_model=19479,model_suite_caches=6,model_examples_total=total,
        selection_sha256=sha(ROOT/'selection.json'),augmentation='none',
        sources_verified_unchanged=True,training_started=False,gpu_used=False,caches=reports))
    save(ROOT/'pipeline-status.json',dict(status='completed',jobs=state,started=started,
        finished=time.time(),training_started=False,summary=str(OUT/'cache-ready.json')))
    return 0


if __name__=='__main__':sys.exit(run())
