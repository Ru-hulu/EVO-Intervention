from libero_sim._mujoco_env import prepare_mujoco_runtime
prepare_mujoco_runtime()
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
import numpy as np
from PIL import Image
suite=benchmark.get_benchmark_dict()['libero_10']();task=suite.get_task(0)
print('TASK',task.name,task.language,flush=True)
p=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
env=OffScreenRenderEnv(bddl_file_name=str(p),camera_heights=256,camera_widths=256)
obs=env.reset();obs,reward,done,info=env.step(np.array([0,0,0,0,0,0,-1.]))
image=obs['agentview_image'];assert image.shape==(256,256,3)
Image.fromarray(image[::-1]).save('/cache/robot-baselines/eval-results/libero-preflight.png')
print('LIBERO_RENDER_STEP_OK',image.shape,image.std(),flush=True)
env.close()
