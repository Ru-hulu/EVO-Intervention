#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
source /cache/robot-baselines/setup/graphics.sh
r=/cache/robot-baselines
export PYBIN=$r/envs/simplemem-rmbench/bin/python
export CHECKPOINT=$r/models/SimpleMemVLA/RMBench
export GPUS=1 NUM_GPUS=1 SEEDS_PER_TASK=1 GROUP_SIZE=1
export TASKS=observe_and_pickup ATTN_IMPLEMENTATION=flash_attention_2
export RUN_DIR=$r/eval-results/simplemem-rmbench
cd "$r/repos/SimpleMemVLA"
bash scripts/eval_rmbench.sh
