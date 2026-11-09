from pathlib import Path
import importlib,os,yaml,numpy as np
from PIL import Image
def main():
    root=Path.cwd();config=root/'env_cfg/task_config'
    args=yaml.safe_load((config/'demo_clean.yml').read_text())
    emb=yaml.safe_load((config/'_embodiment_config.yml').read_text())
    cam=yaml.safe_load((config/'_camera_config.yml').read_text())[args['camera']['head_camera_type']]
    args.update(task_name='observe_and_pickup',task_config='demo_clean',head_camera_h=cam['h'],head_camera_w=cam['w'],dual_arm_embodied=True,eval_mode=True,collect_data=False,eval_video_log=False,save_data=False)
    path=emb[args['embodiment'][0]]['file_path']
    args.update(left_robot_file=path,right_robot_file=path,left_embodiment_config=yaml.safe_load((Path(path)/'config.yml').read_text()),right_embodiment_config=yaml.safe_load((Path(path)/'config.yml').read_text()))
    env=importlib.import_module('envs.observe_and_pickup').observe_and_pickup()
    env.setup_demo(now_ep_num=0,seed=100000,is_test=True,**args)
    obs=env.get_obs();image=obs['observation']['head_camera']['rgb'];assert image.shape[-1]==3 and image.std()>1
    Image.fromarray(image).save('/cache/robot-baselines/eval-results/rmbench-official-task.png')
    print('OFFICIAL_RMBENCH_TASK_RESET_RENDER_OK',image.shape,float(image.std()),flush=True)
    env.take_action(np.asarray(obs['joint_action']['vector']),action_type='qpos')
    print('OFFICIAL_RMBENCH_TASK_ACTION_OK',flush=True)
    env.close_env()

if __name__ == "__main__":
    main()
