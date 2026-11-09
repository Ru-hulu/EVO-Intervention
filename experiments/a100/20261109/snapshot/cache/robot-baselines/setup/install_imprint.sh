#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.file:///cache/robot-baselines/dependencies/imprint-lerobot.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/huggingface/lerobot
export UV_PROJECT_ENVIRONMENT=$r/envs/imprint
cd "$r/repos/imprint"
uv sync --frozen --no-dev --python "$r/python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11"
uv pip check --python "$UV_PROJECT_ENVIRONMENT/bin/python"
echo IMPRINT_INSTALL_DONE
