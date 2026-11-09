#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
p=$r/envs/imprint/bin/python
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.file:///cache/robot-baselines/dependencies/imprint-lerobot.insteadOf
export GIT_CONFIG_VALUE_0=https://github.com/huggingface/lerobot
cd "$r/repos/imprint"
uv pip install --python "$p" setuptools==80.9.0 hatchling poetry-core pip wheel editables
uv pip install --python "$p" --no-deps --no-build-isolation -r "$r/setup/imprint-locked-local-wheels.txt" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
uv pip check --python "$p"
echo IMPRINT_LOCKED_MIRROR_INSTALL_DONE
