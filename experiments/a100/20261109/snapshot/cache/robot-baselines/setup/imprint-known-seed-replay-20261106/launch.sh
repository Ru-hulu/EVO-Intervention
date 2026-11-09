#!/bin/bash
source /cache/robot-baselines/setup/common.sh
source /cache/robot-baselines/setup/graphics.sh
export IMPRINT_BATCH_ROOT=/cache/robot-baselines IMPRINT_RESULTS_BATCH=imprint-known-seed-replay-20261106 IMPRINT_HOST=Huawei IMPRINT_SEED=230000
export VLA_WORKER_MODEL=recurrent VLA_WORKER_GPU=3
exec /cache/robot-baselines/envs/imprint/bin/python -u /cache/robot-baselines/setup/imprint-known-seed-replay-20261106/collect.py
