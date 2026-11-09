
import datetime, fcntl, hashlib, json, os, subprocess, sys, traceback
from pathlib import Path

setup = Path('/cache/cvpzhr/setup/robobrain25-forward-20261105')
config_path = setup/'config.json'
config = json.loads(config_path.read_text())
output = Path(config['output_dir'])
state_path = setup/'training-status.json'
def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(state):
    tmp = state_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(state_path)
lock = (setup/'training.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state = dict(status='starting', started_at=now(), supervisor_pid=os.getpid(),
             config_path=str(config_path), config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
             output_dir=str(output), log_path=str(setup/'train.log'),
             gpu_indices=[0,1,2,3], expected_updates=9967, validation_steps=[4984,9967])
save(state)
try:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Refuse to overwrite a nonempty formal output directory')
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:
        raise RuntimeError('GPUs acquired another compute process before launch: '+active)
    command = ['/cache/cvpzhr/envs/robometer/bin/python','-m','torch.distributed.run',
               '--standalone','--nproc_per_node=4',
               '/cache/cvpzhr/code/robobrain25-forward-20261105/train.py',
               '--config',str(config_path)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    state['command'] = command
    print(json.dumps(dict(event='launch',time=now(),command=command)),flush=True)
    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                             cwd='/cache/cvpzhr/code/robobrain25-forward-20261105')
    state.update(status='running',torchrun_pid=child.pid)
    save(state)
    returncode = child.wait()
    state.update(status='completed' if returncode == 0 else 'failed',
                 finished_at=now(), returncode=returncode)
    for name in ['trainer_state.json','last.json','best.json']:
        p = output/name
        if p.exists():
            obj = json.loads(p.read_text())
            if name == 'trainer_state.json':
                obj = {k:obj.get(k) for k in ['global_step','epoch','best_metric','best_model_checkpoint']}
            state[name] = obj
    save(state)
    print(json.dumps(dict(event=state['status'],time=now(),returncode=returncode)),flush=True)
except BaseException:
    state.update(status='failed', finished_at=now(), error=traceback.format_exc())
    save(state)
    traceback.print_exc()
    sys.exit(1)
