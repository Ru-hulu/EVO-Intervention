# Usage: source /cache/robot-baselines/setup/activate-robomme-policy.sh
source /cache/robot-baselines/envs/robomme-policy/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="/cache/robot-baselines/repos/robomme_policy_learning/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENPI_DATA_HOME=/cache/robot-baselines/.cache/openpi
export XDG_CACHE_HOME=/cache/robot-baselines/.cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
cd /cache/robot-baselines/repos/robomme_policy_learning
