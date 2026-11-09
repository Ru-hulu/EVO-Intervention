#!/bin/bash
source /cache/robot-baselines/setup/common.sh
source /cache/robot-baselines/setup/graphics.sh
r=/cache/robot-baselines
s=$r/setup/proprimem-extra-benchmarks-20261104
export LIBERO_CONFIG_PATH=$s/libero-mem-config
export MUJOCO_EGL_DEVICE_ID=3 CUDA_VISIBLE_DEVICES=3
$r/envs/simplemem-libero/bin/python "$s/prepare_mem_states.py"
export MUJOCO_EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2
cd "$r/repos/SimpleMemVLA"
exec "$r/envs/simplemem-libero/bin/python" "$s/libero_mem_entry.py" --pretrained_checkpoint "$r/models/SimpleMemVLA/LIBERO" --tasks libero_mem/0 libero_mem/1 libero_mem/2 libero_mem/3 libero_mem/4 libero_mem/5 libero_mem/6 libero_mem/7 --episodes_per_task 10 --group_size 1 --num_gpus 1 --attn_implementation sdpa --video_dir "$r/eval-results/proprimem-extra-20261104/simplemem-libero-mem/videos" --video_max_per_task 10 --log_file "$r/eval-results/proprimem-extra-20261104/simplemem-libero-mem/eval.log"
