#!/usr/bin/env bash
source /cache/robot-baselines/setup/common.sh
source /cache/robot-baselines/setup/graphics.sh
r=/cache/robot-baselines
x=$r/repos/RoboDojo/XPolicyLab
out=$r/eval-results/simplemem-robodojo
mkdir -p "$out"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
cd "$x"
CUDA_VISIBLE_DEVICES=2 "$r/envs/simplemem-robodojo/bin/python" setup_policy_server.py \
 --config_path policy/SimpleMemVLA/deploy.yml --overrides \
 port=8038 host=127.0.0.1 bench_name=RoboDojo task_name=stack_bowls \
 ckpt_name=downloaded env_cfg_type=arx_x5 seed=0 policy_name=SimpleMemVLA action_type=joint \
 checkpoint_path="$r/models/SimpleMemVLA/RoboDojo/ckpt/RoboDojo/SimpleMemVLA/RoboDojo-sim-arx_x5-joint-0" \
 attn_implementation=flash_attention_2 > "$out/policy.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
"$r/envs/simplemem-robodojo/bin/python" - "$server" <<'PY'
import sys,os,time
from websockets.sync.client import connect
pid=int(sys.argv[1])
for _ in range(900):
 try:
  with connect('ws://127.0.0.1:8038',open_timeout=1):break
 except OSError:
  os.kill(pid,0);time.sleep(1)
else:raise TimeoutError('RoboDojo policy startup exceeded 900 seconds')
PY
cd "$r/repos/RoboDojo"
export PYTHONPATH=$PWD:$x
export EVAL_NUM=1
export ROBODOJO_RUN_ID=dependency-smoke-20261104
"$r/envs/robodojo/bin/python" -u src/eval_client/main.py \
 --task_name stack_bowls --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
 --device cuda:3 --device_id 3 --policy_name SimpleMemVLA --port 8038 --host 127.0.0.1 \
 --additional_info ckpt_name=downloaded,action_type=joint --seed 0 --headless \
 --kit_args '--/renderer/multiGpu/enabled=false --/renderer/activeGpu=3 --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera' \
 > "$out/simulator.log" 2>&1
