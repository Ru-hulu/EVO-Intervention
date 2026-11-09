#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="/cache/cvpzhr/setup/forest-original-cap50-train-20261104"
PYTHON="/cache/cvpzhr/envs/forest/bin/python"
export PYTHONPATH="$TASK_ROOT/code"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
TASK_NPROC="${NPROC_PER_NODE:-4}"
cd "$TASK_ROOT"
exec 9>"$TASK_ROOT/launch.lock"
flock -n 9
case "${1:-prepare}" in
  prepare)
    "$PYTHON" "$TASK_ROOT/prepare_train.py" prepare
    ;;
  smoke)
    "$PYTHON" "$TASK_ROOT/prepare_train.py" prepare
    test ! -e "$TASK_ROOT/smoke-output"
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_NPROC" \
      -m mem_lite.train_stage3 --config "$TASK_ROOT/smoke.config.json" \
      2>&1 | tee "$TASK_ROOT/smoke.log"
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_NPROC" \
      -m mem_lite.train_stage3 --resume "$TASK_ROOT/smoke-output/last.pt" --max-steps 3 \
      2>&1 | tee "$TASK_ROOT/smoke-resume.log"
    "$PYTHON" "$TASK_ROOT/prepare_train.py" finish-smoke --world-size "$TASK_NPROC"
    ;;
  train)
    "$PYTHON" "$TASK_ROOT/prepare_train.py" check-launch --world-size "$TASK_NPROC"
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$TASK_NPROC" \
      -m mem_lite.train_stage3 --config "$TASK_ROOT/config.json" \
      2>&1 | tee "$TASK_ROOT/train.log"
    ;;
  *)
    echo "Usage: $0 {prepare|smoke|train}" >&2
    exit 2
    ;;
esac
