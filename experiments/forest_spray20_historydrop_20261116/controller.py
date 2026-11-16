"""Single detached four-GPU job with checkpoint-at-step resource protection."""
import fcntl
import os
import subprocess
import time
import traceback
from .common import *
from experiments.recover_validation_20261114.common import host_sample, safe
from experiments.soc35_finetune_20261114.queue import idle
from experiments.recover_spray20_train_20261116.queue import session_children


def main():
    lock = (ROOT/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'launch.json').exists(), 'Inspect existing job; never double launch'
    assert not (ROOT/'resource-stop.json').exists()
    assert not OUT.exists() or not any(OUT.iterdir()), 'Fresh output required'
    save(ROOT/'controller.pid', os.getpid())
    verify_code(); verify_bindings()
    resources = host_sample(); baseline = resources['oom_kill']
    assert idle([0,1]) and idle([2,3]), 'All four GPUs must be free; no other jobs interrupted'
    assert all(safe(resources, i, baseline) for i in range(4)), resources
    command = [PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', '-m', PACKAGE+'.train']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', NCCL_DEBUG='WARN', TRITON_CACHE_DIR=str(ROOT/'triton-cache'))
    for k in ('PRM_CUDA_MEMORY_FRACTION', 'FOREST_NATIVE_GATED_NORM'): env.pop(k, None)
    with (ROOT/'train.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT/'code', env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(ROOT/'launch.json', dict(pid=child.pid, command=command, started=time.time(), initial_resources=resources))
    while child.poll() is None:
        resources = host_sample()
        okay = all(safe(resources, i, baseline) for i in range(4))
        if not okay and not (ROOT/'resource-stop.json').exists():
            save(ROOT/'resource-stop.json', dict(reason='Resource reserve reached', resources=resources, updated=time.time()))
        logs = rows(OUT/'train.jsonl') if (OUT/'train.jsonl').exists() else []
        save(ROOT/'status.json', dict(status='training' if okay else 'stopping_at_checkpoint',
            controller_pid=os.getpid(), train_pid=child.pid, children=session_children(child.pid),
            gpus=[0,1,2,3], latest=logs[-1] if logs else None, resources=resources, updated=time.time(),
            expected_steps=read(ROOT/'plan.json')['expected_steps'], evaluation_started=False))
        time.sleep(5)
    save(ROOT/'exit.json', dict(returncode=child.returncode, finished=time.time()))
    deadline = time.time()+120
    while session_children(child.pid):
        assert time.time() < deadline, 'Old training children remain alive'
        time.sleep(2)
    assert child.returncode == 0, 'Training failed; all artifacts retained'
    if (ROOT/'paused.json').exists():
        save(ROOT/'status.json', dict(status='resource_paused', checkpoint=str(OUT/'last.pt'), updated=time.time()))
        return
    proof = read(ROOT/'checkpoint-verification.json'); assert proof['status'] == 'passed'
    save(ROOT/'status.json', dict(status='completed', verification=proof, updated=time.time(), evaluation_started=False))


if __name__ == '__main__':
    try: main()
    except BaseException:
        error = dict(error=traceback.format_exc(), updated=time.time())
        save(ROOT/'controller-error.json', error)
        status = read(ROOT/'status.json') if (ROOT/'status.json').exists() else {}
        status.update(status='failed', **error); save(ROOT/'status.json', status)
        raise
