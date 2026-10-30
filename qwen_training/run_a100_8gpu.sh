#!/usr/bin/env bash
# 13图窗口八卡入口。首参数为输出目录，必须通过 --index-files 指定新格式索引。
# batch 是未经性能验证的起始配置，可通过后续参数覆盖。
set -euo pipefail
source /data/evo-inter/env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

EVO_RUN_DIR=${1:-/data/evo-inter/outputs/reference-8gpu-$(date +%Y%m%d-%H%M%S)}
if (( $# )); then shift; fi
cd /data/evo-inter/code

exec python -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m qwen_training.train_distributed \
  --model-path /data/evo-inter/models/Qwen3.5-4B \
  --output-dir "$EVO_RUN_DIR" \
  --global-batch-size 16 --val-batch-size 1 \
  --num-workers 4 --prefetch-factor 2 \
  --gradient-checkpointing --require-fast-kernels \
  --max-pixels 262144 --lr 1e-4 \
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
  --epochs 12 --save-epochs 2 4 6 8 10 12 --log-every 50 \
  "$@"
