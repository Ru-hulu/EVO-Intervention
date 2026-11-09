#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/streamforest-recover-code
export OMP_NUM_THREADS=4
exec /root/autodl-tmp/envs/mem-lite/bin/python -m mem_lite.datasets.prepare_recovery cache \
  --prepared-dir /root/autodl-tmp/streamforest-data/insert-screw-recover \
  --model-path /root/autodl-tmp/models/Qwen3.5-4B \
  --cache-dir /root/autodl-tmp/streamforest-cache/insert-screw-recover \
  --sample-fps 5 --device cuda
