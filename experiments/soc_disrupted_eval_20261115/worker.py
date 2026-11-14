"""Controller-ticketed worker; no import-time inference or implicit launches."""
import argparse
import fcntl
import os
import time
import traceback
from .common import *
from .guard import WorkerGuard, process_identity, host_sample, gpu_processes, idle_safe
from .engine import run_episode, resume_canary, SafetyPause


def run(job_id, token):
    lock = (ROOT/'jobs'/f'{job_id}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard = WorkerGuard(job_id, token)
    source = read(ROOT/'inputs.json'); input_sha = sha(ROOT/'inputs.json')
    assert input_sha == guard.ticket['inputs_sha256']
    assert source['source_code_sha256'] == code_hashes()
    job = guard.ticket['job']; info = source['registry'][f"{job['model']}-{job['suite']}"]
    path = ROOT/'jobs'/f'{job_id}.json'
    state = dict(job=job, job_id=job_id, inputs_sha256=input_sha, process=process_identity(os.getpid()),
        controller=guard.ticket['controller'], gpu=guard.gpu, status='loading', updated=time.time())
    save(path, state)
    def update(**values):
        state.update(values, updated=time.time()); save(path, state)
    try:
        import random
        import numpy as np
        import torch
        # Importing torch can take seconds. Recheck the idle slot immediately
        # before creating our CUDA context, then reserve it before model loading.
        if not idle_safe(host_sample(),gpu_processes(),guard.gpu,guard.ticket['initial_oom']):
            raise SafetyPause('GPU became occupied during imports')
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.70)
        torch.cuda.manual_seed_all(42)
        guard.last=0;guard.check()
        from .backend import Backend
        backend = Backend(info)
        guard.check()
        save(ROOT/'jobs'/f'{job_id}-initialization.json', backend.audit)
        update(status='running', initialization_sha256=digest(backend.audit))
        if job['phase'] == 'canary':
            checks = []
            for condition in ('static_only-seed42', 'rewind_only-seed42'):
                candidates = task_inputs(source, dict(job, shard=0, shards=1), condition)
                row, entry = min(candidates, key=lambda p:len(p[1]['target_pt']))
                directory = ROOT/'canary'/f"{job['model']}-{job['suite']}"/condition
                checks.append(dict(condition=condition, key=row['key'], comparisons=resume_canary(
                    backend, row, entry, identity(input_sha, info, row, entry), directory, guard.check)))
            proof = dict(status='passed', inputs_sha256=input_sha, initialization=backend.audit,
                checks=checks, gpu_forward_tested=True, formal_predictions_used=False)
            save(ROOT/'canary'/f"{job['model']}-{job['suite']}.json", proof)
        else:
            proof = read(ROOT/'canary'/f"{job['model']}-{job['suite']}.json")
            assert proof['status']=='passed' and proof['inputs_sha256']==input_sha and proof['gpu_forward_tested']
            assert proof['initialization']==backend.audit
            completed = 0
            for condition in CONDITIONS:
                for row,entry in task_inputs(source, job, condition):
                    guard.check()
                    update(condition=condition, episode=row['key'], completed_versions=completed)
                    from experiments.prm_five_groups_20261114.stream_images import verify_cache
                    verify_cache(Path(row['image_root']), row,
                        dict(video_hashes=row['video_hashes'], frame_indices=row['frame_indices']), False)
                    run_episode(backend, row, entry, identity(input_sha,info,row,entry),
                        result_path(job['model'],condition,row['key']), guard.check,
                        lambda n:update(episode_observations=n))
                    completed += 1
            update(completed_versions=completed)
        update(status='completed')
        return 0
    except SafetyPause as e:
        update(status='paused', reason=str(e)); return 75
    except Exception:
        update(status='failed', error=traceback.format_exc()); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--job', required=True); p.add_argument('--ticket', required=True)
    args=p.parse_args(); raise SystemExit(run(args.job,args.ticket))
