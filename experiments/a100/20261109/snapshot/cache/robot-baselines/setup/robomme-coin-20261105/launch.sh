#!/usr/bin/env bash
set -euo pipefail
source /cache/robot-baselines/setup/activate-robomme-policy.sh
export PYTHONPATH=/cache/robot-baselines/setup/robomme-coin-20261105:$PYTHONPATH
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export PYTHONUNBUFFERED=1
python /cache/robot-baselines/setup/robomme-coin-20261105/features.py
python /cache/robot-baselines/setup/robomme-coin-20261105/train_coin.py
