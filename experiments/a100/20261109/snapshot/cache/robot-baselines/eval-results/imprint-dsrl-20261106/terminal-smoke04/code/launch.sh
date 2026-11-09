#!/usr/bin/env bash
set -euo pipefail
source /cache/robot-baselines/setup/common.sh
source /cache/robot-baselines/setup/graphics.sh
exec /cache/robot-baselines/envs/imprint-dsrl/bin/python -u /cache/robot-baselines/setup/imprint-dsrl-20261106/launch.py "$@"
