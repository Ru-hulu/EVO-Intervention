#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
p=$r/envs/simplemem-rmbench/bin/python
uv pip install --python "$p" --no-build-isolation deepspeed==0.16.9 einops 'numpy<2' 'setuptools<81' wheel pybind11 setuptools-scm
c=$r/repos/SimpleMemVLA/rmbench_sim/rmbench
cp -an "$r/assets/RMBench_sim/assets" "$c/"
cp -an "$r/assets/RMBench_sim/envs/curobo" "$c/envs/"
uv pip install --python "$p" --no-build-isolation -e "$c/envs/curobo"
uv pip check --python "$p"
echo RMBENCH_READY
