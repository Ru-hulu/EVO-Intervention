#!/usr/bin/env bash
set -euo pipefail
RBM_ENV=/root/envs/robometer
RBM_UV=/root/miniconda3/bin/uv
RBM_REV=352d160389daa964788de1ec933d1925f3a6de4f
RBM_SOURCE="/root/robometer/third_party/robometer-$RBM_REV"
RBM_HERE=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2
export UV_CACHE_DIR=/root/.cache/robometer/uv PIP_CACHE_DIR=/root/.cache/robometer/pip
export HF_HOME=/root/.cache/robometer/huggingface TMPDIR=/root/.cache/robometer/tmp
export UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=2 UV_HTTP_TIMEOUT=30
mkdir -p "$TMPDIR"
if [ ! -e "$RBM_ENV/bin/python" ]; then
    "$RBM_UV" venv --python /root/miniconda3/bin/python3.12 "$RBM_ENV"
fi
"$RBM_ENV/bin/python" "$RBM_HERE/seed_runtime.py"
"$RBM_UV" pip install --verbose --python "$RBM_ENV/bin/python" \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r "$RBM_HERE/requirements.txt"
mkdir -p /root/robometer/third_party
if [ ! -d "$RBM_SOURCE/.git" ]; then
    source /etc/network_turbo >/dev/null 2>&1
    git clone --filter=blob:none --no-checkout https://github.com/robometer/robometer.git "$RBM_SOURCE"
    git -C "$RBM_SOURCE" sparse-checkout set robometer
    git -C "$RBM_SOURCE" checkout --detach "$RBM_REV"
fi
test "$(git -C "$RBM_SOURCE" rev-parse HEAD)" = "$RBM_REV"
"$RBM_UV" pip freeze --python "$RBM_ENV/bin/python" > "$RBM_HERE/environment.lock.txt"
"$RBM_UV" pip check --python "$RBM_ENV/bin/python"
PYTHONPATH="$RBM_SOURCE" "$RBM_ENV/bin/python" -c \
    'import torch, transformers, peft; from robometer.models.rbm import RBM; print("Imports OK:", torch.__version__, transformers.__version__, peft.__version__)'
echo ENVIRONMENT_SETUP_COMPLETED
