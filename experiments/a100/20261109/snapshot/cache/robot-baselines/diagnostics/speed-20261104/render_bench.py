from pathlib import Path
import sys,time,json,subprocess,os
import numpy as np
OUT=Path(__file__).parent

def main(preset):
 from rmbench_sim.rmbench_env import RMBenchSimEnv
 import sapien
 if preset=='rt8':
  original=sapien.render.set_ray_tracing_samples_per_pixel
  sapien.render.set_ray_tracing_samples_per_pixel=lambda _:original(8)
 elif preset=='raster':
  original=sapien.render.set_camera_shader_dir
  sapien.render.set_camera_shader_dir=lambda _:original('default')
 started=time.perf_counter();e=RMBenchSimEnv('press_button','demo_clean','/cache/robot-baselines/dependencies/EventVLA/RoboTwin-Mem')
 try:
  e.reset(40,instruction='Observe the two numbers on the table.');setup=time.perf_counter()-started
  for _ in range(2):e.get_obs();e.TASK_ENV.cameras.get_third_view_rgb()
  timings=[];extra=[]
  for i in range(8):
   t=time.perf_counter();obs=e.get_obs();timings.append(time.perf_counter()-t)
   t=time.perf_counter();e.TASK_ENV.cameras.get_third_view_rgb();extra.append(time.perf_counter()-t)
   if i==0:np.savez(OUT/(preset+'-images.npz'),**{k:obs['observation'][k]['rgb'] for k in ['head_camera','left_camera','right_camera']})
  result=dict(preset=preset,setup_seconds=setup,get_obs_seconds=float(np.median(timings)),extra_third_seconds=float(np.median(extra)),samples=8,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),note='Static scene, no model/physics steps; shared GPU 3. Raster/rt8 change appearance and are diagnostic only.')
  (OUT/(preset+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 finally:e.close()
if __name__=='__main__':
 if len(sys.argv)>1:main(sys.argv[1])
 else:
  for preset in ['rt32','rt8','raster']:subprocess.run([sys.executable,__file__,preset],check=True)
