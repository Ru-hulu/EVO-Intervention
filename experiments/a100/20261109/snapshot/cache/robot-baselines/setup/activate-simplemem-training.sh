# Usage: source /cache/robot-baselines/setup/activate-simplemem-training.sh
# Reuse the existing RMBench-compatible training dependencies without changing other envs.
source /cache/robot-baselines/envs/simplemem-rmbench/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="/cache/robot-baselines/repos/SimpleMemVLA${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export BACKBONE=/cache/cvpzhr/models/Qwen3.5-4B
cd /cache/robot-baselines/repos/SimpleMemVLA
