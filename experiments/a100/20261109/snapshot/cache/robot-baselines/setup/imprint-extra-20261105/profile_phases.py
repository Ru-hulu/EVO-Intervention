"""Passive inclusive wall-clock timing; preserves calls, arguments and results."""
import atexit,json,time
from collections import defaultdict
from pathlib import Path
stats=defaultdict(lambda:[0,0.])
def install():
 import capture
 from envs._base_task import Base_Task
 from envs.camera import Camera
 def wrap(cls,name,label):
  old=getattr(cls,name)
  def timed(*a,**kw):
   started=time.perf_counter()
   try:return old(*a,**kw)
   finally:
    stats[label][0]+=1;stats[label][1]+=time.perf_counter()-started
  setattr(cls,name,timed)
 for cls,name,label in [(Base_Task,'get_obs','observation_total'),(Base_Task,'take_action','action_total'),(Base_Task,'_update_render','scene_render_update'),(Camera,'update_picture','camera_render'),(Camera,'get_rgb','camera_readback'),(capture.Session,'observe','record_video_and_state')]:wrap(cls,name,label)
 old=capture.Session.close
 def close(self):
  started=time.perf_counter();old(self);stats['record_close'][0]+=1;stats['record_close'][1]+=time.perf_counter()-started
  (self.out/'timing.json').write_text(json.dumps(dict(inclusive_wall_seconds=True,phases=dict(stats)),indent=2))
 capture.Session.close=close
