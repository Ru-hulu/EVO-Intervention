#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Evo-Inter
PROC_ENV=/root/autodl-tmp/envs/procvlm
UV=/root/miniconda3/bin/uv
REV=377523a31f05bab9c0db5ac8b9edfa7b7f03968a
UPSTREAM="$ROOT/third_party/ProcVLM-$REV"
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2
export UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=2
export UV_HTTP_TIMEOUT=180

if [ ! -e "$PROC_ENV/bin/python" ]; then
    "$UV" venv --python /root/miniconda3/bin/python3.12 "$PROC_ENV"
fi
"$PROC_ENV/bin/python" "$HERE/seed_runtime.py"
"$UV" pip install --python "$PROC_ENV/bin/python" \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r "$HERE/requirements.txt"

mkdir -p "$ROOT/third_party"
if [ ! -d "$UPSTREAM/.git" ]; then
    source /etc/network_turbo >/dev/null 2>&1
    git clone --filter=blob:none --no-checkout https://github.com/RUCKBReasoning/ProcVLM.git "$UPSTREAM"
    git -C "$UPSTREAM" sparse-checkout set evqa core configs
    git -C "$UPSTREAM" checkout --detach "$REV"
fi
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = "$REV"
"$PROC_ENV/bin/python" "$HERE/patch_upstream.py" "$UPSTREAM"
"$UV" pip freeze --python "$PROC_ENV/bin/python" > "$HERE/environment.lock.txt"
PYTHONPATH="$UPSTREAM" "$PROC_ENV/bin/python" -c \
    'import torch, transformers, peft; from evqa.model import ProcVLMWithValueHead; print("Imports OK:", torch.__version__, transformers.__version__, peft.__version__)'
echo 'ENVIRONMENT_SETUP_COMPLETED'
