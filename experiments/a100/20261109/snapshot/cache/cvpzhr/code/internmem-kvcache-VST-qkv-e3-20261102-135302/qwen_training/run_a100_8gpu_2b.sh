#!/usr/bin/env bash
# Qwen3.5-2B，13图窗口；新配置尚未进行性能验证。
# Usage: bash run_a100_8gpu_2b.sh OUTPUT_DIR --index-files WINDOWS.jsonl [other arguments...]
set -euo pipefail
EVO_2B_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EVO_2B_RUN_DIR=${1:-/data/evo-inter/outputs/reference-2b-8gpu-$(date +%Y%m%d-%H%M%S)}
if (( $# )); then shift; fi
exec bash "$EVO_2B_SCRIPT_DIR/run_a100_8gpu.sh" "$EVO_2B_RUN_DIR" \
  --model-path /data/evo-inter/models/Qwen3.5-2B \
  --global-batch-size 16 --val-batch-size 1 \
  --num-workers 6 --prefetch-factor 2 \
  --epochs 10 --save-epochs 2 4 6 8 10 \
  "$@"
