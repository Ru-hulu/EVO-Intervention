import sapien.core as sapien
import numpy as np
from PIL import Image
engine=sapien.Engine();renderer=sapien.SapienRenderer();engine.set_renderer(renderer)
scene=engine.create_scene();scene.set_timestep(1/240);scene.add_ground(0)
scene.set_ambient_light([.5,.5,.5]);scene.add_directional_light([0,1,-1],[1,1,1])
b=scene.create_actor_builder();b.add_box_collision(half_size=[.1,.1,.1]);b.add_box_visual(half_size=[.1,.1,.1]);actor=b.build();actor.set_pose(sapien.Pose([0,0,.1]))
cam=scene.add_camera('test',128,128,1.,.01,10);cam.set_pose(sapien.Pose([-1,0,.5],[.973,0,.23,0]))
scene.step();scene.update_render();cam.take_picture();frame=cam.get_picture('Color');assert np.isfinite(frame).all();Image.fromarray((frame[:,:,:3].clip(0,1)*255).astype('uint8')).save('/cache/robot-baselines/eval-results/sapien-preflight.png');print('SAPIEN_RENDER_STEP_OK',frame.shape,float(frame.std()),flush=True)
