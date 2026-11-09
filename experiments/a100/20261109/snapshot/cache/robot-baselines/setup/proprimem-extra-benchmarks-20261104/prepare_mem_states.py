from pathlib import Path
import os,sys,json,yaml
import numpy as np
import torch
r=Path('/cache/robot-baselines');s=Path(__file__).parent;mem=r/'repos/LIBERO-Mem';dest=s/'generated-init-states';dest.mkdir(exist_ok=True)
p=s/'libero-mem-config/config.yaml';cfg=yaml.safe_load(p.read_text());cfg['init_states']=str(dest);p.write_text(yaml.safe_dump(cfg));os.environ['LIBERO_CONFIG_PATH']=str(p.parent)
for root in [mem/'thirdparty/robomimic',mem/'thirdparty/robosuite',mem]:sys.path.insert(0,str(root))
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite=benchmark.get_benchmark_dict()['libero_mem']();report=[]
for index in range(8):
 task=suite.get_task(index);target=dest/task.problem_folder/task.init_states_file
 if target.exists():continue
 env=OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),camera_heights=256,camera_widths=256,controller='OSC_POSE',horizon=2600)
 try:
  states=[]
  for seed in range(10):env.seed(10000+seed);env.reset();states.append(env.get_sim_state())
  target.parent.mkdir(exist_ok=True);torch.save(np.asarray(states),target);report.append(dict(task=task.name,states=10,seeds=list(range(10000,10010)),provenance='Fresh native environment resets; not original public demo initial states'))
 finally:env.close()
(s/'init-state-provenance.json').write_text(json.dumps(report,indent=2))
