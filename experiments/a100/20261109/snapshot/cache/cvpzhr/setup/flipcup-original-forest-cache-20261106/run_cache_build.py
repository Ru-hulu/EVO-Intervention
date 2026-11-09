"""Detached cache construction and validation only; never launches training."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SETUP = Path(__file__).resolve().parent
PYTHON = '/cache/cvpzhr/envs/forest/bin/python'
MODULE = 'mem_lite.datasets.prepare_flipcup_forest'


def main():
    lock = (SETUP / 'cache-build.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((SETUP / 'plan.json').read_text())
    cache = Path(plan['cache_dir'])
    env = dict(os.environ, PYTHONPATH=str(SETUP / 'code'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    started = time.time()
    state = dict(pid=os.getpid(), started_at=started, cache_dir=str(cache), code=str(SETUP/'code'),
                 worker_count=plan['workers'], gpu_ids=plan['gpu_ids'], training_started=False)

    def status(phase, **extra):
        state.update(phase=phase, updated_at=time.time(), elapsed_seconds=time.time()-started, **extra)
        temporary = SETUP / 'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(SETUP/'status.json')

    def stage(command, arguments, gpu):
        status(command)
        with (SETUP/f'{command}.log').open('a') as log:
            result = subprocess.run([PYTHON, '-u', '-m', MODULE, command, *arguments],
                                    cwd=SETUP/'code', env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status(command+'_failed', returncode=result.returncode)
            return False
        return True

    if not stage('prepare', ['--plan', str(SETUP/'plan.json')], ''):
        return 1
    if not stage('smoke', ['--cache-dir', str(cache)], str(plan['gpu_ids'][0])):
        return 1
    processes = []
    for i in range(plan['workers']):
        with (SETUP/f'worker-{i}.log').open('a') as log:
            process = subprocess.Popen([PYTHON, '-u', '-m', MODULE, 'worker', '--cache-dir', str(cache),
                                        '--worker-id', str(i), '--workers', str(plan['workers'])],
                                       cwd=SETUP/'code', env=dict(env, CUDA_VISIBLE_DEVICES=str(plan['gpu_ids'][i % len(plan['gpu_ids'])])),
                                       stdout=log, stderr=subprocess.STDOUT)
        processes.append(process)
    status('building', worker_pids=[p.pid for p in processes])
    while any(p.poll() is None for p in processes):
        status('building', returncodes=[p.poll() for p in processes])
        time.sleep(10)
    if any(p.returncode for p in processes):
        status('build_failed', returncodes=[p.returncode for p in processes])
        return 1
    if not stage('validate', ['--cache-dir', str(cache)], ''):
        return 1
    status('completed', validation_report=str(cache/'validation-report.json'),
           returncodes=[p.returncode for p in processes])
    return 0


if __name__ == '__main__':
    sys.exit(main())
