from pathlib import Path
import importlib.metadata as metadata
import json
import os
import subprocess

root=Path('/cache/cvpzhr')
python=root/'envs/forest/bin/python'
source='/root/autodl-tmp/streamforest-cache/rebuild-20261030'
data=root/'datasets/forest-5fps'
config=json.loads((root/'setup/reference-train.config.json').read_text())
config['model_path']=str(root/'models/Qwen3.5-4B')
config['sample_files']=[p.replace(source,str(data),1) for p in config['sample_files']]
config.update(batch_size=2,gradient_accumulation_steps=1,num_workers=2,persistent_workers=True,
              output_dir=str(root/'outputs/forest-4gpu'))
(root/'setup/train-4gpu.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
selected=[next(p for p in config['sample_files'] if '/memlong/' in p),next(p for p in config['sample_files'] if '/insert-screw-recover/' in p)]
(root/'setup/preflight-samples.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
for dist in metadata.distributions():
    for entry in dist.entry_points:
        if entry.group!='console_scripts': continue
        path=python.parent/entry.name
        path.write_text(f'#!{python}\nimport sys\nfrom importlib.metadata import distribution\ne=next(e for e in distribution({dist.metadata["Name"]!r}).entry_points if e.group=="console_scripts" and e.name=={entry.name!r})\nsys.exit(e.load()())\n')
        path.chmod(0o755)
freeze=subprocess.check_output([str(python),'-m','pip','freeze','--all'],text=True)
(root/'setup/requirements.lock.txt').write_text(freeze)
environment='''export PATH=/cache/cvpzhr/envs/forest/bin:$PATH
export HF_HOME=/cache/cvpzhr/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/cache/cvpzhr/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/cache/cvpzhr/cache/torchinductor
export XDG_CACHE_HOME=/cache/cvpzhr/cache
export PIP_CACHE_DIR=/cache/cvpzhr/cache/pip
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export PYTHONPATH=/cache/cvpzhr/code/forest
'''
(root/'setup/environment.sh').write_text(environment)
launcher='''#!/usr/bin/env bash
set -euo pipefail
source /cache/cvpzhr/setup/environment.sh
cd /cache/cvpzhr/code/forest
python -c 'import json; from pathlib import Path; assert json.loads(Path("/cache/cvpzhr/setup/readiness.json").read_text())["status"] == "PASS", "Environment checks are incomplete"'
exec python -m torch.distributed.run --standalone --nproc_per_node=4 -m mem_lite.train_stage3 --config /cache/cvpzhr/setup/train-4gpu.json "$@"
'''
(root/'setup/train-4gpu.sh').write_text(launcher)
(root/'setup/train-4gpu.sh').chmod(0o755)
print('Configured isolated environment and four-GPU launcher; training not started')
