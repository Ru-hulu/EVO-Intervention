from pathlib import Path
import shutil,subprocess
r=Path('/cache/robot-baselines')
for target,src,updater in [
 (r/'repos/RMBench',r/'assets/RMBench_official','scripts/update_embodiment_config_path.py'),
 (r/'dependencies/EventVLA/RoboTwin-Mem',r/'assets/IMPRINT_RMBench/assets','script/update_embodiment_config_path.py')]:
 for part in ['embodiments','objects']:
  shutil.copytree(src/part,target/'assets'/part,dirs_exist_ok=True)
 subprocess.run([str(r/'envs/rmbench/bin/python'),updater],cwd=target,check=True)
 print('ASSETS_CONFIG_READY',target,flush=True)
for name in ['rmbench','simplemem-rmbench']:
 site=r/'envs'/name/'lib/python3.10/site-packages'
 p=site/'sapien/wrapper/urdf_loader.py';s=p.read_text();s=s.replace('open(urdf_file, "r")','open(urdf_file, "r", encoding="utf-8")').replace('urdf_file[:-4] + "srdf"','urdf_file[:-4] + ".srdf"');p.write_text(s)
 p=site/'mplib/planner.py';s=p.read_text().replace('if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:','if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:');p.write_text(s)
 print('UPSTREAM_COMPAT_PATCHES',name,flush=True)
