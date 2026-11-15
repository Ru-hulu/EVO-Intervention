"""Bounded CPU-only cache preparation; no training or evaluation launches."""
import fcntl
import subprocess
import sys
import time
from .common import *


def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'controller.pid').write_text(str(os.getpid())+'\n')
    assert not (OUT/'final-audit.json').exists(), 'Already complete'
    for m in MODELS: assert read(ROOT/f'{m}-State-status.json')['status']=='smoke_passed'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    jobs=[]
    # Six workers, two CPU threads each; processors only, no vision/model forwards.
    for model in MODELS:
        for suite in SUITES:
            resource_check()
            with (ROOT/f'{model}-{suite}.log').open('a') as log:
                child=subprocess.Popen([PYTHON,'-m','experiments.soc35_grm_caches_20261115.build',
                    '--model',model,'--suite',suite],cwd=ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((model,suite,child))
            time.sleep(1)
    while True:
        state=[dict(model=m,suite=s,pid=p.pid,returncode=p.poll()) for m,s,p in jobs]
        pending=any(r['returncode'] is None for r in state)
        failed=[r for r in state if r['returncode'] not in (None,0)]
        save(ROOT/'status.json',dict(status='preparing' if pending else 'failed' if failed else 'auditing',
             jobs=state,failures=failed,updated=time.time(),training_started=False,gpu_used=False))
        if not pending:break
        time.sleep(10)
    if failed:return
    result=subprocess.run([PYTHON,'-m','experiments.soc35_grm_caches_20261115.audit'],cwd=ROOT/'code',env=env)
    save(ROOT/'status.json',dict(status='completed' if result.returncode==0 else 'audit_failed',jobs=state,
         updated=time.time(),training_started=False,gpu_used=False,audit_returncode=result.returncode,
         final_audit=str(OUT/'final-audit.json')))


if __name__=='__main__':main()
