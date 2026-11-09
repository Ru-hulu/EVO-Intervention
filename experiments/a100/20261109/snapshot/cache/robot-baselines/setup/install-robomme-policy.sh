#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
p=$r/envs/robomme-policy/bin/python
uv venv --allow-existing --python "$r/python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11" "$r/envs/robomme-policy"
uv pip sync --python "$p" "$r/setup/robomme-runtime-locked.txt"
uv pip install --python "$p" --no-deps -e "$r/repos/robomme_policy_learning" -e "$r/repos/robomme_policy_learning/packages/openpi-client"
# Imported unconditionally by the upstream PyTorch model, even for JAX inference.
uv pip install --python "$p" pytest==8.3.5
uv pip check --python "$p"
uv pip freeze --python "$p" > "$r/manifests/installed-robomme-policy.txt"
