"""Start one authorized 10-epoch training run and record its eventual exit status."""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/evo-inter')
AUDIT = ROOT / 'verification/qwen35_2b_training_10ep_20261010'
TUNING = ROOT / 'verification/qwen35_2b_tuning_20261010'
AUDIT.mkdir(parents=True, exist_ok=True)
lock = (AUDIT / 'active.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if (AUDIT / 'launch.json').exists():
    raise RuntimeError('This authorized run already has a launch record; inspect it before any restart.')
verified = json.loads((TUNING / 'final_report.json').read_text())
assert verified['status'] == 'complete' and verified['two_short_train_validation_cycles_passed']
for name, expected in verified['provenance']['production_code_sha256'].items():
    path = ROOT / 'code/qwen_training' / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Tested code changed: {path}'
model = ROOT / 'models/Qwen3.5-2B'
download = ROOT / 'verification/qwen35_2b_20261010'
assert json.loads((download / 'download_result.json').read_text())['status'] == 'complete'
manifest = json.loads((download / 'modelscope_manifest.json').read_text())['Data']['Files']
for item in manifest:
    if item['Type'] == 'blob':
        assert (model / item['Path']).stat().st_size == item['Size']
indices = [Path('/root/dataset/RW-RL-Dataset/piperx_insert_copper_screw/samples/samples.jsonl'), Path('/root/dataset/piperx-bolt-intentional-drop-reinsert-22ep-raw/samples/samples.jsonl')]
assert all(path.is_file() for path in indices)
busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
if busy:
    raise RuntimeError('Existing GPU compute processes; refusing to start a duplicate job.')
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
available_gib = int(mem['MemAvailable'].split()[0]) / 2**20
assert available_gib > 100
assert shutil.disk_usage(ROOT).free > 20 * 2**30
launcher = ROOT / 'code/qwen_training/run_a100_8gpu_2b.sh'
subprocess.run(['bash', '-n', str(launcher)], check=True)
stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%SZ')
output = ROOT / 'outputs' / ('qwen35-2b-8gpu-10ep-' + stamp)
output.mkdir(parents=True, exist_ok=False)
command = ['bash', str(launcher), str(output), '--epochs', '10', '--save-epochs', '2', '4', '6', '8', '10']
env = os.environ.copy()
for key in ['NCCL_P2P_LEVEL', 'NCCL_P2P_DISABLE']:
    env.pop(key, None)
with (output / 'train.log').open('x') as log:
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT / 'code'), env=env, start_new_session=True, close_fds=True)
    launch = {'status': 'running', 'formal_training': True, 'started_unix': time.time(), 'started_utc': datetime.now(timezone.utc).isoformat(), 'driver_pid': os.getpid(), 'torchrun_pid': child.pid, 'process_group_id': child.pid, 'command': command, 'output_dir': str(output), 'log_path': str(output / 'train.log'), 'model_path': str(model), 'world_size': 8, 'per_gpu_batch': 8, 'global_batch_size': 64, 'num_workers_per_gpu': 6, 'val_batch_size_per_gpu': 16, 'epochs': 10, 'save_epochs': [2, 4, 6, 8, 10], 'resume': None, 'host_available_gib_at_launch': available_gib, 'launcher_sha256': hashlib.sha256(launcher.read_bytes()).hexdigest(), 'tested_source_sha256': verified['provenance']['production_code_sha256']}
    for path in [AUDIT / 'launch.json', output / 'launch.json']:
        path.write_text(json.dumps(launch, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(launch, ensure_ascii=False), flush=True)
    code = child.wait()
result = {**launch, 'status': 'complete' if code == 0 else 'failed', 'exit_code': code, 'finished_unix': time.time()}
for path in [AUDIT / 'result.json', output / 'result.json']:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'event': 'training_exited', 'exit_code': code, 'output_dir': str(output)}), flush=True)
sys.exit(code)
