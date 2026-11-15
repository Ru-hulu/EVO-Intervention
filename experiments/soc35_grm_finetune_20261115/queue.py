"""One detached queue per GPU pair; measured pilots, no unknown-error retries."""
import argparse
import fcntl
import os
import statistics
import subprocess
import time
import traceback
from .common import *
from experiments.soc35_finetune_20261114.queue import idle, wait_idle
from experiments.soc_disrupted_eval_20261115.guard import host_sample
from experiments.recover_validation_20261114.common import safe


def health(gpus, baseline):
    sample = host_sample()
    return all(safe(sample, gpu, baseline) for gpu in gpus), sample


def launch(cfg, pilot, status_path, baseline):
    assert idle(cfg['gpus']), 'Unrelated CUDA context present'
    okay, sample = health(cfg['gpus'], baseline); assert okay, sample
    config_path = Path(cfg['setup_dir'])/'config.json'; save(config_path, cfg)
    folder = Path(cfg['setup_dir'])/('pilot' if pilot else 'formal'); folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'launch.json').exists(), 'Refuse duplicate launch'
    command = [cfg['python'], '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               '-m', PACKAGE+'.train', '--config', str(config_path)]
    if pilot: command.append('--pilot')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpus'])), OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1', PYTORCH_ALLOC_CONF='expandable_segments:True',
        NCCL_DEBUG='WARN', TRITON_CACHE_DIR=str(ROOT/cfg['name']/'triton-cache'))
    for k in ('PRM_CUDA_MEMORY_FRACTION', 'FOREST_NATIVE_GATED_NORM'): env.pop(k,None)
    with (folder/'train.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT/'code', env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(folder/'launch.json', dict(pid=child.pid, command=command, gpus=cfg['gpus'], started=time.time(),
        config_sha256=sha(config_path)))
    while child.poll() is None:
        okay, sample = health(cfg['gpus'], baseline)
        status = read(status_path); status.update(active_pid=child.pid, updated=time.time(), resources=sample)
        save(status_path, status)
        if not okay: save(ROOT/cfg['name']/'resource-stop.json', dict(reason='Resource reserve reached', resources=sample))
        time.sleep(5)
    save(folder/'exit.json', dict(returncode=child.returncode, finished=time.time()))
    wait_idle(cfg['gpus'])
    return child.returncode, folder


def main(model):
    verify_code()
    home = ROOT/model; home.mkdir(parents=True, exist_ok=True)
    lock = (home/'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    status_path = home/'status.json'
    assert not status_path.exists(), 'Inspect existing queue before any recovery'
    configs = read(ROOT/'plan.json')['configs'][model]; gpus = GPUS[model]
    baseline = host_sample()['oom_kill']
    save(home/'controller.pid',os.getpid())
    save(status_path, dict(status='waiting_for_idle_cards', model=model, gpus=gpus, controller_pid=os.getpid(),
        started=time.time(),updated=time.time(),completed=[],order=SUITES))
    stable = None
    while True:
        okay, sample = health(gpus, baseline)
        if idle(gpus) and okay:
            stable = stable or time.time()
            if time.time()-stable >= 15:break
        else:stable = None
        status = read(status_path); status.update(updated=time.time(),resources=sample);save(status_path,status)
        time.sleep(5)
    trials = []
    # Start from the proven Recovery configuration; only expand based on measured reserve.
    candidates = [(4,False), (8,True)]
    for micro, gc in candidates:
        cfg = configure_batch(configs['State'],micro,gc)
        cfg['setup_dir'] = str(home/'trials'/f'b{micro}-gc{int(gc)}')
        status = read(status_path);status.update(status='calibrating',candidate=dict(microbatch=micro,gradient_checkpointing=gc));save(status_path,status)
        code, folder = launch(cfg,True,status_path,baseline)
        assert not (home/'resource-stop.json').exists(), 'Resource protection triggered; inspect before recovery'
        if code == 0:
            proof = read(folder/'verification.json');assert proof['status']=='passed' and proof['pilot']
            elapsed = max(statistics.median(v[1:]) for v in proof['pilot_step_seconds'])
            trials.append(dict(status='passed',microbatch=micro,gradient_checkpointing=gc,
                samples_per_second=2*micro/elapsed,peak_reserved_gib=proof['peak_reserved_gib'],verification=str(folder/'verification.json')))
            # 8 without recomputation is tried only if baseline's conservative scaling fits.
            if micro == 4 and proof['peak_reserved_gib']*2 < 60:candidates.append((8,False))
        else:
            log=(folder/'train.log').read_text().lower()
            assert 'out of memory' in log or 'gpu headroom' in log, f'Unknown pilot error; inspect {folder}'
            trials.append(dict(status='memory_limit',microbatch=micro,gradient_checkpointing=gc,log=str(folder/'train.log')))
        save(home/'trials.json', trials)
    choice = choose_trial(trials)
    save(home/'batch-choice.json',dict(choice=choice,fixed_effective_batch=16,pilot_weights_discarded=True,
        policy='Best measured post-warmup throughput among bounded native-input configurations'))
    for suite in SUITES:
        cfg=configure_batch(configs[suite],choice['microbatch'],choice['gradient_checkpointing'])
        status=read(status_path);status.update(status='training',suite=suite,microbatch=cfg['microbatch'],
            gradient_checkpointing=cfg['gradient_checkpointing'],effective_batch=16,updated=time.time());save(status_path,status)
        code,folder=launch(cfg,False,status_path,baseline)
        assert code==0, f'{suite} failed; preserve checkpoint and inspect {folder}'
        proof=read(folder/'verification.json')
        assert proof['status']=='passed' and not proof['pilot'] and proof['epoch']==1
        assert proof['expected_episodes']==35 and proof['expected_samples']==cfg['samples']
        assert proof['selection_sha256']==cfg['selection_sha256'] and proof['train_input_sha256']==sha(cfg['train_data'])
        status=read(status_path);status['completed'].append(dict(suite=suite,verification=proof))
        status.update(active_pid=None,updated=time.time());save(status_path,status)
    status.update(status='completed',finished=time.time());save(status_path,status)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=MODELS);a=p.parse_args()
    try:main(a.model)
    except BaseException:
        path=ROOT/a.model/'status.json';d=read(path) if path.exists() else {}
        d.update(status='failed',error=traceback.format_exc(),updated=time.time());save(path,d)
        raise
