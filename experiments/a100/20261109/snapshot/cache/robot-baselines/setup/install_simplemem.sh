#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
family=$1
p=$r/envs/simplemem-$family/bin/python
uv venv --python /usr/bin/python3.10 "$r/envs/simplemem-$family"
# DeepSpeed is training-only and is not needed by the evaluation entrypoints.
sed '/^deepspeed/d' "$r/repos/SimpleMemVLA/requirements.txt" > "$r/setup/simplemem-eval-requirements.txt"
uv pip install --python "$p" -r "$r/setup/simplemem-eval-requirements.txt" pip setuptools wheel ninja packaging
uv pip install --python "$p" --no-deps -e "$r/repos/SimpleMemVLA"
if [ "$family" = libero ]; then
 uv pip install --python "$p" robosuite==1.4.0 mujoco==2.3.2 bddl==1.0.1 easydict future gym==0.25.2 cloudpickle hydra-core thop
 site=$($p -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
 echo "$r/repos/LIBERO" > "$site/zz_libero_repo.pth"
else
 uv pip install --python "$p" -r "$r/repos/SimpleMemVLA/requirements/rmbench_sim.txt" 'numpy<2' 'zarr<3' 'moviepy<2'
fi
uv pip check --python "$p"
echo INSTALL_DONE_$family
