#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^[0-7]$ ]]; then
  echo "Usage: bash run_both.sh <free GPU index 0..7>" >&2
  exit 2
fi
benchmark_dir=$(cd "$(dirname "$0")" && pwd)
benchmark_gpu=$1
source /data/evo-inter/env.sh
for benchmark_size in 2B 4B; do
  /data/evo-inter/.venv/bin/python -u "$benchmark_dir/benchmark.py" \
    --output "$benchmark_dir" --gpu "$benchmark_gpu" \
    --model-size "$benchmark_size" --warmups 2 --runs 5 \
    2>&1 | tee "$benchmark_dir/run_${benchmark_size}.log"
done
