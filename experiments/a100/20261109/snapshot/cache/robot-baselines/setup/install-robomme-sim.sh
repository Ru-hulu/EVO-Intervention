#!/usr/bin/env bash
set -euo pipefail
source /cache/robot-baselines/setup/common.sh
export UV_PROJECT_ENVIRONMENT=/cache/robot-baselines/envs/robomme-sim
export GIT_CONFIG_COUNT=2 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0='' GIT_CONFIG_KEY_1=https.proxy GIT_CONFIG_VALUE_1=''
cd /cache/robot-baselines/repos/RoboMME
uv sync --frozen --python /cache/robot-baselines/envs/robomme-policy/bin/python
