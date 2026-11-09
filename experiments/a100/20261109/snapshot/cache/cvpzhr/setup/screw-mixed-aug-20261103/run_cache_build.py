"""Sixteen cache workers on four GPUs, followed by complete CPU validation; no training."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

SETUP = Path('/cache/cvpzhr/setup/screw-mixed-aug-20261103')
CACHE = Path('/cache/cvpzhr/datasets/forest-screw-mixed-aug-5fps-20261103')
CODE = '/cache/cvpzhr/code/screw-mixed-aug-cache-20261103'
PYTHON = '/cache/cvpzhr/envs/forest/bin/python'
WORKERS = 16


def status(value):
    value.update(updated_at=time.time(), training_started=False)
    temp = SETUP/'cache-build-status.partial.json'
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(SETUP/'cache-build-status.json')


def main():
    import fcntl
    lock = (SETUP/'cache-build.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = dict(os.environ, PYTHONPATH=CODE, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    processes = []
    started = time.time()
    for i in range(WORKERS):
        log = (SETUP/f'cache-worker-{i}.log').open('a')
        process = subprocess.Popen([PYTHON, '-m', 'mem_lite.datasets.prepare_screw_mixed', 'worker',
                                    '--cache-dir', str(CACHE), '--worker-id', str(i), '--workers', str(WORKERS)],
                                   env=dict(environment, CUDA_VISIBLE_DEVICES=str(i%4)), stdout=log, stderr=subprocess.STDOUT)
        log.close()
        processes.append(process)
    state = dict(pid=os.getpid(), started_at=started, phase='building',
                 worker_pids=[p.pid for p in processes], code=CODE, cache=str(CACHE))
    while any(p.poll() is None for p in processes):
        state['returncodes'] = [p.poll() for p in processes]
        status(state)
        time.sleep(10)
    state['returncodes'] = [p.returncode for p in processes]
    if any(p.returncode for p in processes):
        state['phase'] = 'failed'
        status(state)
        return 1
    state['phase'] = 'validating'
    status(state)
    with (SETUP/'cache-validation.log').open('w') as log:
        check = subprocess.run([PYTHON, '-m', 'mem_lite.datasets.prepare_screw_mixed', 'validate',
                                '--cache-dir', str(CACHE), '--workers', '8'],
                               env=dict(environment, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT)
    state.update(phase='completed' if check.returncode==0 else 'validation_failed',
                 validation_returncode=check.returncode, elapsed_seconds=time.time()-started)
    status(state)
    return check.returncode


if __name__=='__main__':
    sys.exit(main())
