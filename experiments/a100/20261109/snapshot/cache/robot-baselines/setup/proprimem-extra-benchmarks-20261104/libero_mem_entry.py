import os,sys,runpy
from pathlib import Path
r=Path('/cache/robot-baselines');mem=r/'repos/LIBERO-Mem';repo=r/'repos/SimpleMemVLA'
for p in [repo,mem/'thirdparty/robomimic',mem/'thirdparty/robosuite',mem]:sys.path.insert(0,str(p))
import libero.libero.benchmark
import libero_sim.libero_env as env
env.SUITES.append('libero_mem');env.TASK_SUITE_MAX_STEPS['libero_mem']=2500
if __name__=='__main__':runpy.run_module('libero_sim.eval_success',run_name='__main__')
