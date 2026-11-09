from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':128,'height':128,'anti_aliasing':0})
print('ISAAC_APP_OPENED',flush=True)
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.rotations import euler_angles_to_quat
world=World(stage_units_in_meters=1.)
world.scene.add_default_ground_plane()
world.scene.add(DynamicCuboid(prim_path='/World/cube',position=np.array([0,0,.5]),scale=np.array([.4,.4,.4]),color=np.array([1.,0.,0.])))
cam=Camera(prim_path='/World/camera',position=np.array([-2.,0.,1.]),resolution=(128,128),orientation=euler_angles_to_quat(np.array([0.,20.,0.]),degrees=True))
world.reset();cam.initialize()
for _ in range(40):world.step(render=True)
img=cam.get_rgba()
assert img.shape==(128,128,4),img.shape
from PIL import Image
Image.fromarray(img).save('/cache/robot-baselines/eval-results/isaac-preflight.png')
print('ISAAC_RENDER_STEP_OK',img.shape,float(img[:,:,:3].std()),flush=True)
app.close()
