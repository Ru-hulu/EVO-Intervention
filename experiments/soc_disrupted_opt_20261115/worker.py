"""Controller-ticketed worker; no import-time inference or implicit launches."""
import argparse
import fcntl
import os
import time
import traceback
from .common import *
from experiments.soc_disrupted_eval_20261115.guard import WorkerGuard, process_identity, host_sample, gpu_processes, idle_safe
from .engine import run_episode
from experiments.soc_disrupted_eval_20261115.engine import SafetyPause


def run(job_id, token):
    lock = (ROOT/'jobs'/f'{job_id}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    guard = WorkerGuard(job_id, token)
    source = read(ROOT/'inputs.json'); input_sha = sha(ROOT/'inputs.json')
    assert input_sha == guard.ticket['inputs_sha256']
    assert source['source_code_sha256'] == code_hashes()
    manifest=read(OPT/'runtime-manifest.json')
    assert manifest['files']==runtime_hashes(), 'Optimized runtime changed after release'
    receipt=verify_choice(job_id)
    execution=read(OPT/'execution-configs.json')[job_id]
    assert any(c['accepted'] and c['batch_size']==execution['batch_size'] for c in receipt['candidates'])
    assert execution['tuning_receipt_sha256']==sha(OPT/'tuning'/f'{job_id}.json')
    assert execution['runtime_manifest_sha256']==sha(OPT/'runtime-manifest.json')
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
        backend = Backend(info,batch_size=execution['batch_size'])
        assert backend.audit==receipt['initialization']
        if execution.get('exact_input_memoization'):
            from .memo import Memoized,gpu_gate
            backend=Memoized(backend,True)
            update(stage='exact_input_memoization_gpu_gate')
            memo_path=OPT/'memoization'/f'{job_id}.json'
            if memo_path.exists():
                memo=read(memo_path)
                assert memo['status']=='passed' and memo['repeated_outputs_exact']
                assert memo['initialization_sha256']==digest(backend.audit) and memo['inputs_sha256']==input_sha
                assert memo['code_sha256']==sha(Path(__file__).parent/'memo.py')
            else:gpu_gate(backend,source,job,guard.check)
            execution=dict(execution,memoization_receipt_sha256=sha(memo_path))
        guard.check()
        save(ROOT/'jobs'/f'{job_id}-initialization.json', backend.audit)
        update(status='running', initialization_sha256=digest(backend.audit))
        assert job['phase']=='suite' and job['shard']==0 and job['shards']==1
        canary_path=ROOT/'canary'/f"{job['model']}-{job['suite']}.json"
        validate_canary(read(canary_path),input_sha,backend.audit)
        update(stage='formal',execution=execution)
        completed = 0
        for condition in CONDITIONS:
            for row,entry in task_inputs(source, job, condition):
                guard.check()
                update(condition=condition, episode=row['key'], completed_versions=completed)
                from experiments.prm_five_groups_20261114.stream_images import verify_cache
                verify_cache(Path(row['image_root']), row,
                    dict(video_hashes=row['video_hashes'], frame_indices=row['frame_indices']), False)
                run_episode(backend, row, entry, identity(input_sha,info,row,entry),
                    result_path(job['model'],condition,row['key']), execution, guard.check,
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
