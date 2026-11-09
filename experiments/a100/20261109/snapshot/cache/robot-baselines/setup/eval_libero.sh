#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
source /cache/robot-baselines/setup/graphics.sh
r=/cache/robot-baselines
export LIBERO_CONFIG_PATH=$r/setup/libero-config
export PYBIN=$r/envs/simplemem-libero/bin/python
export CHECKPOINT=$r/models/SimpleMemVLA/LIBERO
export GPUS=0 NUM_GPUS=1 EPISODES_PER_TASK=1 GROUP_SIZE=1
export TASKS=libero_spatial/0 ATTN_IMPLEMENTATION=sdpa
export RUN_DIR=$r/eval-results/simplemem-libero
export VIDEO_MAX_PER_TASK=1
cd "$r/repos/SimpleMemVLA"
bash scripts/eval_libero.sh
