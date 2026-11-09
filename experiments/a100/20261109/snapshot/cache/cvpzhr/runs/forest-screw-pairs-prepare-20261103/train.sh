#!/usr/bin/env bash
set -euo pipefail
cd /cache/cvpzhr/code/forest-screw-pairs-20261103
export CUDA_VISIBLE_DEVICES=1,2,3
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
/cache/cvpzhr/envs/forest/bin/python - <<'PY'
import hashlib,json,pathlib,subprocess
r=pathlib.Path('/cache/cvpzhr/runs/forest-screw-pairs-prepare-20261103')
ready=json.loads((r/'ready.json').read_text())
cfg=json.loads((r/'finetune-config.json').read_text())
assert ready['status']=='ready' and ready['formal_training_started'] is False
assert ready['config_sha256']==hashlib.sha256((r/'finetune-config.json').read_bytes()).hexdigest()
files=json.loads(pathlib.Path(cfg['sample_files']).read_text())
report=json.loads(pathlib.Path(ready['cache_report']).read_text())
assert files==[row['sample_file'] for row in report['episodes'] if row['split']=='train']
for path in files:
    s=json.loads(pathlib.Path(path).with_name('summary.json').read_text())['source']
    assert s['split']=='train' and s['label_status']=='ready'
out=pathlib.Path(cfg['output_dir'])
if out.exists() and any(out.iterdir()):raise RuntimeError('Training output is already nonempty; choose a new run or resume explicitly')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
used={int(a):int(b) for a,b in (line.split(',') for line in gpu.splitlines())}
if any(used[i]>1024 for i in (1,2,3)):raise RuntimeError('GPU 1/2/3 are no longer idle: '+gpu)
print('Ready: 68 paired training sequences, review excluded, new optimizer, GPU 1/2/3',flush=True)
PY
exec /cache/cvpzhr/envs/forest/bin/torchrun --standalone --nproc_per_node=3 \
  -m mem_lite.train_stage3 \
  --config /cache/cvpzhr/runs/forest-screw-pairs-prepare-20261103/finetune-config.json
