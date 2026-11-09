#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
uv venv --python /usr/bin/python3.10 "$r/envs/rmbench"
p=$r/envs/rmbench/bin/python
# azure is a retired meta-package for data description generation; its maintained SDK is installed below via upstream requirements.
sed '/^azure==/d' "$r/repos/RMBench/scripts/requirements.txt" > "$r/setup/rmbench-requirements.txt"
uv pip install --python "$p" -r "$r/setup/rmbench-requirements.txt" 'numpy<2' 'zarr<3' 'moviepy<2' pip wheel 'setuptools<81' ninja pybind11 setuptools-scm numpy-quaternion yourdfpy importlib-resources scikit-image einops
uv pip install --python "$p" --no-build-isolation -e "$r/repos/SimpleMemVLA/rmbench_sim/rmbench/envs/curobo"
uv pip install --python "$p" warp-lang==1.12.0
uv pip check --python "$p"
echo OFFICIAL_RMBENCH_INSTALL_DONE
