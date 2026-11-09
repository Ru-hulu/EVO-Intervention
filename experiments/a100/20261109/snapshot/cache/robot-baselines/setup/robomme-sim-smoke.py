import json,time
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from robomme.env_record_wrapper import BenchmarkEnvBuilder
out=Path('/cache/robot-baselines/eval-results/robomme-sim-smoke');out.mkdir(parents=True,exist_ok=True)
t=time.time()
b=BenchmarkEnvBuilder(env_id='PickXtimes',dataset='test',action_space='joint_angle',gui_render=False,max_steps=8)
e=b.make_env_for_episode(0)
try:
 obs,info=e.reset()
 frames=[]
 def capture(obs):
  a=obs['front_rgb_list'][-1];a=a.cpu().numpy() if hasattr(a,'cpu') else np.asarray(a)
  frames.append(a.astype(np.uint8))
 capture(obs)
 for _ in range(3):
  action=np.concatenate([np.asarray(obs['joint_state_list'][-1]).reshape(-1),np.asarray(obs['gripper_state_list'][-1]).reshape(-1)[:1]])
  obs,reward,terminated,truncated,info=e.step(action)
  capture(obs)
  if terminated or truncated:break
 imageio.imwrite(out/'frame.png',frames[-1])
 imageio.mimsave(out/'smoke.mp4',frames,fps=5)
 (out/'report.json').write_text(json.dumps({'status':'passed','task':'PickXtimes','episode':0,'steps':len(frames)-1,'image_shape':list(frames[-1].shape),'obs_keys':list(obs),'seconds':time.time()-t,'note':'Environment reset/render/hold-position actions only, not policy success evaluation'},indent=2))
 print((out/'report.json').read_text())
finally:e.close()
