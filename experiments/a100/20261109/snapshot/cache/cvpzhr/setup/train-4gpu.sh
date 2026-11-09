#!/usr/bin/env bash
set -euo pipefail
source /cache/cvpzhr/setup/environment.sh
cd /cache/cvpzhr/code/forest
python -c 'import json; from pathlib import Path; assert json.loads(Path("/cache/cvpzhr/setup/readiness.json").read_text())["status"] == "PASS", "Environment checks are incomplete"'
exec python -m torch.distributed.run --standalone --nproc_per_node=4 -m mem_lite.train_stage3 --config /cache/cvpzhr/setup/train-4gpu.json "$@"
