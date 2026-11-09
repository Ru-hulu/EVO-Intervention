#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
uv venv --python /usr/bin/python3.10 "$r/envs/simplemem-robodojo"
p=$r/envs/simplemem-robodojo/bin/python
uv pip install --python "$p" -r "$r/setup/simplemem-eval-requirements.txt" 'transformers==5.13.1' 'tokenizers>=0.22,<0.23' 'opencv-python-headless<4.12' einops pip setuptools wheel ninja
uv pip install --python "$p" --no-deps -e "$r/repos/RoboDojo/XPolicyLab"
echo SIMPLEMEM_ROBODOJO_INSTALL_DONE
