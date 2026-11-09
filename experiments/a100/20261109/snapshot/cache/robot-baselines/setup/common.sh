#!/usr/bin/env bash
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PATH=/home/ma-user/.local/bin:/usr/local/cuda/bin:$PATH
export UV_CACHE_DIR=/cache/robot-baselines/.uv-cache
export UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple
export UV_HTTP_TIMEOUT=120
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8
export XDG_CACHE_HOME=/cache/robot-baselines/.cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
