#!/bin/bash
source /cache/robot-baselines/setup/common.sh
source /cache/robot-baselines/setup/graphics.sh
s=/cache/robot-baselines/setup/imprint-extra-20261105
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export VLA_WORKER_MODEL=$1 VLA_WORKER_GPU=$2
if [ "$1" = simplemem ]; then
 exec /cache/robot-baselines/envs/simplemem-rmbench/bin/python -u "$s/worker.py"
else
 exec /cache/robot-baselines/envs/imprint/bin/python -u "$s/worker.py"
fi
