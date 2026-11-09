#!/usr/bin/env bash
set -euo pipefail
base=/cache/robot-baselines
source "$base/setup/common.sh"
source "$base/setup/activate-simplemem-training.sh"
CUDA_VISIBLE_DEVICES=3 python -u "$base/setup/robomme-eval-20261105/serve_simplemem.py" > "$base/logs/simplemem-robomme-policy.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
source "$base/setup/activate-robomme-sim.sh"
export PYTHONPATH="$base/repos/robomme_policy_learning/examples/robomme:$base/repos/robomme_policy_learning/packages/openpi-client/src"
CUDA_VISIBLE_DEVICES=3 python -u "$base/setup/robomme-eval-20261105/run_simplemem.py"
