"""Passive task-state evidence and synchronized training cameras; no success checker calls."""
import atexit,json,os,time
from pathlib import Path
import numpy as np
SESSION=None
CAMERAS=('head_camera','left_camera','right_camera')
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def snapshot(e):
 row={'action_step':int(e.take_action_cnt),'success':bool(e.eval_success),'max_reward':float(e.max_reward)}
 for k in ('stage_id','press_cnt','press_flag','press_cnt_1','press_cnt_2','press_cnt_check_button','card_id_1','card_id_2','current_state_pointer','target_state_transition','fail_flag','flag','selected_idx','target_pose','target_pose1_p','target_pose2_p','correct_combination','combination_lst','block_y_lim'):
  if hasattr(e,k):
   v=getattr(e,k)
   try:json.dumps(v,default=serial);row[k]=v
   except TypeError:pass
 poses={}
 for k in ('block','block1','block2','block3','battery1','battery2','blocks_lst','mat_lst','final_occupy_mat','mid_mat'):
  v=getattr(e,k,None);vs=v if isinstance(v,(list,tuple)) else [v]
  ps=[dict(p=a.get_pose().p,q=a.get_pose().q) for a in vs if hasattr(a,'get_pose')]
  if ps:poses[k]=ps
 row['poses']=poses
 return row
def start(e,out,seed,fps):
 global SESSION
 close();SESSION=Session(e,Path(out),seed,fps)
def close():
 global SESSION
 if SESSION:SESSION.close();SESSION=None
class Session:
 def __init__(self,e,out,seed,fps):
  import imageio.v2 as imageio
  self.e=e;self.out=out;out.mkdir(parents=True,exist_ok=True);self.frames=0;self.seed=seed;self.fps=fps
  self.writers={k:imageio.get_writer(str(out/(k+'.mp4')),fps=fps,codec='libx264',pixelformat='yuv420p',macro_block_size=1,ffmpeg_params=['-threads','1']) for k in CAMERAS}
  self.trace=(out/'states.jsonl').open('w');self.actions=(out/'actions.jsonl').open('w');self.started=time.time()
 def observe(self,obs):
  for k,w in self.writers.items():w.append_data(np.asarray(obs['observation'][k]['rgb'],dtype=np.uint8))
  row=snapshot(self.e);row['frame']=self.frames;row['joint_state']=obs['joint_action']['vector'];self.trace.write(json.dumps(row,default=serial)+'\n');self.frames+=1
 def close(self):
  for w in self.writers.values():w.close()
  self.trace.close();self.actions.close()
  (self.out/'evidence.json').write_text(json.dumps(dict(seed=self.seed,frames=self.frames,fps=self.fps,seconds=time.time()-self.started,success=bool(self.e.eval_success),steps=int(self.e.take_action_cnt),renderer_spp=int(os.environ['VLA_RENDER_SPP']),state_capture='passive; native checker not called by observer'),indent=2))
def install_base():
 import sapien
 original=sapien.render.set_ray_tracing_samples_per_pixel
 sapien.render.set_ray_tracing_samples_per_pixel=lambda _:original(int(os.environ['VLA_RENDER_SPP']))
 from envs._base_task import Base_Task
 init=Base_Task._init_task_env_;obs=Base_Task.get_obs;action=Base_Task.take_action
 def initialize(self,**kw):
  kw['data_type']={**kw['data_type'],'third_view':False};kw['eval_video_log']=False;kw['eval_video_save_dir']=None
  return init(self,**kw)
 def observe(self,*a,**kw):
  value=obs(self,*a,**kw)
  if SESSION and SESSION.e is self:SESSION.observe(value)
  return value
 def act(self,*a,**kw):
  if SESSION and SESSION.e is self:SESSION.actions.write(json.dumps({'before_frame':SESSION.frames,'args':a,'kwargs':kw},default=serial)+'\n')
  return action(self,*a,**kw)
 Base_Task._init_task_env_=initialize;Base_Task.get_obs=observe;Base_Task.take_action=act
 atexit.register(close)
def install_simplemem():
 from rmbench_sim.rmbench_env import RMBenchSimEnv
 # Import after the vendored environment is on sys.path.
 import rmbench_sim.rmbench_env as mod
 mod._setup_rmbench_path(mod.VENDORED_SIM_DIR);install_base()
 reset=RMBenchSimEnv.reset;dense=RMBenchSimEnv.take_action_dense
 def resetting(self,seed,instruction=None):
  result=reset(self,seed,instruction);start(self.TASK_ENV,os.environ['VLA_CAPTURE_DIR'],seed,16.6666666667);SESSION.observe(result);return result
 def stepping(self,action,substeps=15):
  if SESSION:SESSION.actions.write(json.dumps({'before_frame':SESSION.frames,'action':action},default=serial)+'\n')
  return dense(self,action,substeps)
 RMBenchSimEnv.reset=resetting;RMBenchSimEnv.take_action_dense=stepping
