#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
r=/cache/robot-baselines
for name in simplemem-libero simplemem-rmbench simplemem-robodojo; do
 p=$r/envs/$name/bin/python
 uv pip install --python "$p" 'transformers==5.13.1' 'tokenizers>=0.22,<0.23' einops 'numpy<2'
 uv pip install --python "$p" --no-deps "$r"/setup/flash_attn-*.whl "$r"/setup/causal_conv1d-*.whl flash-linear-attention==0.2.2
 uv pip install --python "$p" --no-build-isolation deepspeed==0.16.9
 done
