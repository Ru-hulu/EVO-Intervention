#!/usr/bin/env bash
set -euo pipefail
mode="${1:---check}"
case "$mode" in --check|--train) ;; *) echo "Usage: $0 [--check|--train]" >&2; exit 2 ;; esac
source /cache/robot-baselines/setup/activate-simplemem-training.sh
set -a
source /cache/robot-baselines/setup/real-robot/training.env
set +a
export SIMPLEMEMVLA_ROBOT_CONFIG=/cache/robot-baselines/setup/real-robot/robot.json
# Fresh Qwen backbone + random DiT; never load a simulation VLA or resume optimizer state.
export BACKBONE=/cache/cvpzhr/models/Qwen3.5-4B
export RESUME=false CUDA_VISIBLE_DEVICES="$GPUS"
python - <<'CHECK'
from pathlib import Path
from simplememvla.benchmarks import get_benchmark
spec = get_benchmark("real_robot")
for name in ("info.json", "stats.json", "tasks.parquet", "subtasks.parquet"):
    path = Path(spec.root) / "meta" / name
    if not path.is_file():
        raise FileNotFoundError(path)
print("Robot configuration and required dataset metadata found:", spec)
print("Initialization: local Qwen3.5-4B + randomly initialized DiT action head")
CHECK
[[ "$mode" == --check ]] && exit 0
exec bash scripts/train.sh real_robot
