#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec "${PYTHON:-/cache/EVO-Intervention/.venv/bin/python}" -m simplememvla.training.train \
  --config simplememvla/training/configs/huawei_full.json "$@"
