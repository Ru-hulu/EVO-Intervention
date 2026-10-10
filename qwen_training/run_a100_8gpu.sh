#!/usr/bin/env bash
# A100 40GB × 8 的独立训练入口。首参数可指定输出目录，其余参数传给训练器。
set -euo pipefail
source /data/evo-inter/env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1

EVO_RUN_DIR=${1:-/data/evo-inter/outputs/qwen35-8gpu-$(date +%Y%m%d-%H%M%S)}
if (( $# )); then shift; fi
cd /data/evo-inter/code

exec python -u -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m qwen_training.train_distributed \
  --model-path /data/evo-inter/models/Qwen3.5-4B \
  --index-files \
    /root/dataset/RW-RL-Dataset/piperx_insert_copper_screw/samples/samples.jsonl \
    /root/dataset/piperx-bolt-intentional-drop-reinsert-22ep-raw/samples/samples.jsonl \
  --output-dir "$EVO_RUN_DIR" \
  --global-batch-size 32 --val-batch-size 16 \
  --num-workers 4 --prefetch-factor 2 \
  --gpu-image-processing --require-fast-kernels \
  --max-pixels 262144 --lr 1e-4 \
  --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
  --text-loss-weight 0.7 --progress-loss-weight 0.3 \
  --epochs 12 --save-epochs 2 4 6 8 10 12 --log-every 50 \
  "$@"
