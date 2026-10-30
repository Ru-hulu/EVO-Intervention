# Source this file to select only the isolated ProcVLM environment and checkout.
export PATH="/root/autodl-tmp/envs/procvlm/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="/root/autodl-tmp/Evo-Inter/third_party/ProcVLM-377523a31f05bab9c0db5ac8b9edfa7b7f03968a:$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export EVQA_DATASET_ROOT="/root/autodl-tmp/Evo-Inter/results/procvlm/7tasks_5fps_w8_20261028"
export VLM_PATH_QWEN_TRAIN="/root/autodl-tmp/models/ProcVLM-2B"
