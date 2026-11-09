from pathlib import Path
import json

def main():
 from rmbench_sim.rmbench_env import RMBenchSimEnv
 e=RMBenchSimEnv('place_block_mat','demo_clean','/cache/robot-baselines/dependencies/EventVLA/RoboTwin-Mem')
 try:
  r=e.find_solvable_seed(40,max_tries=3);assert r is not None
  seed,info=r;e.reset(seed,instruction='Pick up the blocks from the blue mat and place them on the green mat, then put them back on the original mat, starting from left to right.')
  print('S03_OFFICIAL_EXPERT_SOLVABLE',seed,'step_limit',e.step_lim,flush=True)
  Path('/cache/robot-baselines/setup/proprimem-rmbench-20261104/s03-preflight.json').write_text(json.dumps(dict(seed=seed,expert_solvable=True,step_limit=e.step_lim)))
 finally:e.close()
if __name__=='__main__':main()
