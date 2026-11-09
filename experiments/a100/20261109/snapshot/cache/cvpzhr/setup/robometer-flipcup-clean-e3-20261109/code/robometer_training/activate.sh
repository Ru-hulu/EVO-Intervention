# Source to select the isolated runtime and all caches on the system disk.
export PATH="/root/envs/robometer/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="/root/robometer/third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f:/root/robometer-setup-20261028"
export HF_HOME=/root/.cache/robometer/huggingface
export UV_CACHE_DIR=/root/.cache/robometer/uv
export PIP_CACHE_DIR=/root/.cache/robometer/pip
export TMPDIR=/root/.cache/robometer/tmp
export WANDB_MODE=disabled
