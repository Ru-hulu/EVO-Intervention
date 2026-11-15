"""Two independent GPU-pair queues using proven batches; no calibration runs."""
import argparse
import fcntl
import os
import subprocess
import sys
import time
import traceback
from .common import *
from experiments.soc35_finetune_20261114.queue import idle, wait_idle
from experiments.recover_validation_20261114.common import host_sample, safe


def health(gpus, baseline):
    value = host_sample()
    return all(safe(value, gpu, baseline) for gpu in gpus), value


def session_children(sid):
    result = []
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = p.read_text().split(') ', 1)[1].split()
            if fields[0] != 'Z' and int(fields[3]) == sid: result.append(int(p.parent.name))
        except (OSError, ValueError): pass
    return result


def train_one(cfg, status_path, baseline):
    assert idle(cfg['gpus']), 'Assigned GPU pair occupied; no existing process will be interrupted'
    okay, sample = health(cfg['gpus'], baseline); assert okay, sample
    home = Path(cfg['setup_dir']); folder = home/'formal'; folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'launch.json').exists(), 'Refuse duplicate formal launch'
    assert not (home/'resource-stop.json').exists()
    output = Path(cfg['output_dir'])
    assert not output.exists() or not any(output.iterdir()), 'Refuse nonempty training output'
    config_path = home/'config.json'; save(config_path, cfg)
    module = PACKAGE+('.forest' if cfg['name'] == 'forest' else '.train')
    command = [cfg['python'], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               '-m', module, '--config', str(config_path)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, cfg['gpus'])),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True', NCCL_DEBUG='WARN',
        TRITON_CACHE_DIR=str(ROOT/cfg['name']/'triton-cache'))
    for k in ('PRM_CUDA_MEMORY_FRACTION', 'FOREST_NATIVE_GATED_NORM'): env.pop(k, None)
    with (folder/'train.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT/'code', env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(folder/'launch.json', dict(pid=child.pid, command=command, gpus=cfg['gpus'], started=time.time(),
        config_sha256=sha(config_path), formal=True, pilot=False))
    while child.poll() is None:
        okay, sample = health(cfg['gpus'], baseline)
        status = read(status_path); status.update(active_pid=child.pid, updated=time.time(), resources=sample,
            process_session=session_children(child.pid))
        save(status_path, status)
        if not okay:
            save(home/'resource-stop.json', dict(reason='Resource reserve reached', resources=sample))
        time.sleep(5)
    save(folder/'exit.json', dict(returncode=child.returncode, finished=time.time()))
    until = time.time()+120
    while session_children(child.pid):
        assert time.time() < until, 'Old training children still alive; refuse next job'
        time.sleep(2)
    wait_idle(cfg['gpus'])
    assert child.returncode == 0, f"Training failed; preserved artifacts: {folder}"
    proof = read(folder/'verification.json')
    assert proof['status'] == 'passed' and not proof['pilot'] and proof['epoch'] == 1
    assert proof['expected_episodes'] == 20 and proof['expected_samples'] == 6002
    assert proof['selection_sha256'] == cfg['selection_sha256']
    return proof


def pair(pair_id):
    home = ROOT/f'pair-{pair_id}'; home.mkdir(parents=True, exist_ok=True)
    lock = (home/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path = home/'status.json'; assert not status_path.exists(), 'Inspect old queue before recovery'
    verify_code()
    plan = read(ROOT/'plan.json'); gpus = [2*pair_id, 2*pair_id+1]; order = ORDERS[pair_id]
    baseline = host_sample()['oom_kill']
    save(home/'controller.pid', os.getpid())
    status = dict(status='running', controller_pid=os.getpid(), gpus=gpus, order=order, completed=[],
        started=time.time(), updated=time.time(), calibration_skipped_by_user=True)
    save(status_path, status)
    for name in order:
        cfg = plan['configs'][name]
        assert cfg['gpus'] == gpus
        status = read(status_path); status.update(status='training', model=name, updated=time.time(),
            microbatch=cfg['microbatch'], accumulation=cfg['accumulation'], effective_batch=cfg['effective_batch'])
        save(status_path, status)
        proof = train_one(cfg, status_path, baseline)
        status = read(status_path); status['completed'].append(dict(model=name, verification=proof))
        status.update(active_pid=None, updated=time.time()); save(status_path, status)
    status.update(status='completed', finished=time.time()); save(status_path, status)


def controller():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'status.json').exists(), 'Inspect existing controller before recovery'
    save(ROOT/'controller.pid', os.getpid())
    save(ROOT/'status.json', dict(status='verifying_bindings', controller_pid=os.getpid(), updated=time.time()))
    verify_code(); verify_bindings()
    assert idle([0,1]) and idle([2,3]), 'Four GPUs must be free before initial launch'
    children = []
    for i in ORDERS:
        home = ROOT/f'pair-{i}'; home.mkdir(parents=True, exist_ok=True)
        with (home/'controller.log').open('w') as log:
            child = subprocess.Popen([sys.executable, '-m', PACKAGE+'.queue', '--pair', str(i)],
                cwd=ROOT/'code', stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
    while True:
        statuses = {str(i):read(ROOT/f'pair-{i}/status.json') for i in ORDERS if (ROOT/f'pair-{i}/status.json').exists()}
        exits = [p.poll() for p in children]
        finished = all(c is not None for c in exits)
        state = 'completed' if finished and all(c == 0 for c in exits) else ('failed' if finished else 'running')
        save(ROOT/'status.json', dict(status=state, controller_pid=os.getpid(), updated=time.time(), pairs=statuses,
            completed_models=[v['model'] for s in statuses.values() for v in s.get('completed', [])],
            children=[p.pid for p in children], returncodes=exits, evaluation_started=False))
        if finished: break
        time.sleep(5)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pair', type=int, choices=[0,1]); args = p.parse_args()
    try:
        controller() if args.pair is None else pair(args.pair)
    except BaseException:
        home = ROOT if args.pair is None else ROOT/f'pair-{args.pair}'
        path = home/'status.json'; status = read(path) if path.exists() else {}
        status.update(status='failed', error=traceback.format_exc(), updated=time.time()); save(path, status)
        raise
