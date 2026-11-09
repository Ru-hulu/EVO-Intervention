import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

root=Path('/cache/cvpzhr')
setup=root/'setup'
transfer=json.loads((setup/'extract-status.json').read_text())
assert transfer['phase']=='completed'
checks={}
for name in ['nccl-check','gpu-preflight','loader-check','data-check','assets-check']:
    result=json.loads((setup/(name+'.json')).read_text())
    assert result['status']=='PASS',name
    checks[name]=result
pip=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
assert pip.returncode==0,pip.stdout+pip.stderr
testlog=(root/'logs/cpu-tests.log').read_text()
match=re.search(r'(\d+) passed, (\d+) deselected',testlog)
assert match and 'FAILED ' not in testlog
versions={n:metadata.version(n) for n in ['torch','torchvision','transformers','peft','accelerate','av','flash-linear-attention','datasets','fsspec']}
report={'status':'PASS','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),
        'python':sys.version,'environment':str(Path(sys.executable)), 'versions':versions,
        'gpus':4,'effective_batch_size':8,'batch_per_gpu':2,'gradient_accumulation_steps':1,
        'cpu_tests_passed':int(match[1]),'cpu_optimizer_update_test_deselected':int(match[2]),
        'pip_check':pip.stdout.strip(),'train_samples':checks['data-check']['training_samples'],
        'validation_samples':checks['data-check']['validation_samples'],
        'allocated_peak_gib_per_gpu':[x['allocated_peak_gib'] for x in checks['gpu-preflight']['ranks']],
        'free_disk_gib':shutil.disk_usage(root).free/2**30,
        'code_source':'AutoDL:52086 /root/autodl-tmp/streamforest-recover-code snapshot',
        'source_train_file_sha256':json.loads((setup/'source-manifest.json').read_text())['code/mem_lite/train_stage3.py']['sha256'],
        'data_archives_sha256_verified':len(transfer['completed']),
        'data_bytes_transferred_uncompressed':sum(x['raw_bytes'] for x in transfer['completed']),
        'data_bytes_transferred_compressed':sum(x['compressed_bytes'] for x in transfer['completed']),
        'parameter_updates':0,'training_started':False,
        'launch_command':'bash /cache/cvpzhr/setup/train-4gpu.sh',
        'configuration':str(setup/'train-4gpu.json'),
        'dependency_adjustment':'Added missing datasets dependency; fsspec 2026.9.0 -> 2026.7.0 to satisfy datasets. Copied Python development headers for Triton compilation. Original reference environment unchanged.',
        'notes':['Preflight is a functional startup check, not a sustained throughput benchmark.',
                 'No formal training, optimizer update, shutdown, or edits to shared environments.',
                 'Raw source videos are not required by the transferred image/history caches and were not duplicated.',
                 'The example configuration retains the reference 1 epoch; choose the actual training schedule before launching.']}
(setup/'readiness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(setup/'requirements.lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True))
text=f'''Forest A100 4-GPU environment

Root: {root}
Python: {sys.executable}
Code: {root}/code/forest
Base weights: {root}/models/Qwen3.5-4B
Prepared 5 FPS training/validation data: {root}/datasets/forest-5fps
Training config: {setup}/train-4gpu.json
Validation paths: {setup}/validation.sample_files.json
Logs and check reports: {root}/logs and {setup}

Use environment: source {setup}/environment.sh
Start training when authorized: bash {setup}/train-4gpu.sh
Current example: 4 GPUs x batch 2 x accumulation 1 = effective batch 8; lr=1e-4, BF16,
LoRA rank16/alpha32, gradient checkpointing, one epoch inherited from reference config.
Checkpoint policy is inherited from the reference: latest checkpoint and epoch archives.
No training has been started during setup.
Change train-4gpu.json or pass --epochs N for the intended training schedule.

The original base weights are loaded; no old fine-tuned checkpoint is resumed.
Data includes MemLong and insert-screw-recover, with existing train/validation split.
Raw-video paths in cache summaries are source provenance, not local dependencies.
Use the launcher above; copied historical helper scripts may contain AutoDL paths.
Model/code hashes match the reference snapshot; rsync verified transferred data blocks.
Additional datasets dependency and compatible fsspec version were installed only here.
All caches are directed inside {root}/cache; the machine's /opt/venv is unchanged.
'''
(setup/'README.txt').write_text(text)
print(json.dumps(report,ensure_ascii=False))
