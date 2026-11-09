#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
uv venv --python "$r/python/cpython-3.11.18-linux-x86_64-gnu/bin/python3.11" "$r/envs/robodojo"
p=$r/envs/robodojo/bin/python
uv pip install --python "$p" pip setuptools wheel ninja 'numpy==1.26.0' 'typing_extensions==4.12.2' 'filelock==3.13.1'
uv pip install --python "$p" torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$p" 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.cn --index-strategy unsafe-best-match
uv pip install --python "$p" -r "$r/repos/RoboDojo/scripts/requirements.txt" opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn 'numpy==1.26.0' warp-lang==1.11.0 'onnx>=1.18,<1.22' 'ipython<9' virtualenv==20.30.0
export OMNI_KIT_ACCEPT_EULA=YES
for pkg in "$r"/repos/RoboDojo/third_party/IsaacLab/source/*; do
 [ -f "$pkg/setup.py" ] && uv pip install --python "$p" --no-deps -e "$pkg"
done
echo ROBODOJO_INSTALL_DONE
