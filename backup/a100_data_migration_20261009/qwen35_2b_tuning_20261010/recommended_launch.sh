#!/usr/bin/env bash
# Qwen3.5-2B configuration selected by bounded 8 x A100 tests on 2026-10-10.
# This file was saved and syntax-checked only; formal training has NOT been started.
# Usage: bash run_a100_8gpu_2b.sh [output_directory] [additional_training_arguments...]
# Start from the 2B base model. A 4B checkpoint is not compatible with this backbone.
set -euo pipefail
EVO_2B_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVO_2B_RUN_DIR=${1:-/data/evo-inter/outputs/qwen35-2b-8gpu-$(date +%Y%m%d-%H%M%S)}
if (( $# )); then shift; fi
exec bash "$EVO_2B_SCRIPT_DIR/run_a100_8gpu.sh" "$EVO_2B_RUN_DIR" \
  --model-path /data/evo-inter/models/Qwen3.5-2B \
  --global-batch-size 64 --val-batch-size 16 \
  --num-workers 6 --prefetch-factor 2 \
  "$@"
