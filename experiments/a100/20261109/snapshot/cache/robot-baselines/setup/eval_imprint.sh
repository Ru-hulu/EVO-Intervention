#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
source /cache/robot-baselines/setup/graphics.sh
r=/cache/robot-baselines
profile=$1
export OPENPI_DATA_HOME=$r/.cache/openpi
export HF_HOME=$r/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$r/repos/imprint"
"$r/envs/imprint/bin/python" benchmarks/rmbench/eval/run_local.py \
 --rmbench-root "$r/dependencies/EventVLA/RoboTwin-Mem" \
 --checkpoint "$r/models/IMPRINT/simulation/rmbench/$profile/stage2/step-100000" \
 --dinov3-model "$r/models/DINOv3-ViT-B16" \
 --architecture "$profile" --task observe_and_pickup --trials 1 --start-seed 100000 \
 --output "$r/eval-results/imprint-$profile" \
 --policy-python "$r/envs/imprint/bin/python" --sim-python "$r/envs/rmbench/bin/python" \
 --policy-gpu 0 --sim-gpu 0 --port "${IMPRINT_PORT:-8039}"
