"""Recover lid/tool geometry; refuse labels when saved-state replay disagrees."""
import os,sys,json,hashlib,time
from pathlib import Path
import numpy as np
from transforms3d.quaternions import quat2mat
BASE=Path('/cache/robot-baselines');OUT=BASE/'annotations/s01-wrong-lid-20261105'
sys.path.insert(0,str(BASE/'repos/SimpleMemVLA'))
from rmbench_sim.rmbench_env import RMBenchSimEnv

def main():
 ds,key=sys.argv[1:3];row=next(json.loads(l) for l in (BASE/'viewer-datasets'/ds/'metadata.jsonl').read_text().splitlines() if json.loads(l)['sample_id']==key)
 evidence=Path(row['evidence_path']);states=[json.loads(l) for l in (evidence/'states.jsonl').read_text().splitlines()];actions=[json.loads(l) for l in (evidence/'actions.jsonl').read_text().splitlines()]
 simple='simplemem' in key
 sim=BASE/('repos/SimpleMemVLA/rmbench_sim/rmbench' if simple else 'dependencies/EventVLA/RoboTwin-Mem')
 env=RMBenchSimEnv('cover_blocks','demo_clean',str(sim));env.args['data_type']['third_view']=False;env.args['eval_video_log']=False;env.args['eval_video_save_dir']=None
 e=env.TASK_ENV
 def qpos():return np.asarray(e.robot.get_left_arm_jointState()+e.robot.get_right_arm_jointState())
 e.get_obs=lambda:{'joint_action':{'vector':qpos()}}
 env.reset(row['actual_seed'],row.get('task_prompt'))
 e._update_render=lambda:None
 result=[];maximum=0.;start=time.time()
 def observe():
  nonlocal maximum
  f=len(result);old=states[f];gap=float(np.max(np.abs(qpos()-old['joint_state'])));maximum=max(maximum,gap)
  assert gap<1e-4,(f,'qpos mismatch',gap)
  for name in ['current_state_pointer','fail_flag','success']:
   value=bool(e.eval_success) if name=='success' else getattr(e,name)
   assert value==old[name],(f,name,value,old[name])
  tools=[list(e.get_arm_pose(arm)) for arm in ['left','right']]
  centers=[np.asarray(tool[:3])+quat2mat(tool[3:])@np.array([.12,0,0]) for tool in tools]
  contacts=[np.asarray(c.get_contact_point(0)[:3]) for c in e.covers]
  distances=[[float(np.linalg.norm(center-contact)) for contact in contacts] for center in centers]
  result.append(dict(frame=f,phase=int(e.current_state_pointer),fail=bool(e.fail_flag),covers=[c.get_pose().p.tolist() for c in e.covers],blocks=[b.get_pose().p.tolist() for b in e.blocks],tool_to_grasp_distance=distances,tool_poses=tools))
 try:
  for a in actions:
   while len(result)<a['before_frame']:observe()
   if simple:env.take_action_dense(np.asarray(a['action']),substeps=15)
   else:e.take_action(np.asarray(a['args'][0]),**a.get('kwargs',{}))
  while len(result)<len(states):observe()
  out=OUT/ds/'geometry'/(key+'.json');out.parent.mkdir(parents=True,exist_ok=True)
  out.write_text(json.dumps(dict(sample_id=key,frames=result,target_state_transition=e.target_state_transition,open_order=e.open_lst,max_qpos_error=maximum,source_actions_sha256=hashlib.sha256((evidence/'actions.jsonl').read_bytes()).hexdigest(),seconds=time.time()-start)))
  print('VERIFIED_GEOMETRY',key,len(result),maximum,time.time()-start,flush=True)
 finally:
  robots=[getattr(e.robot,side+'_proc',None) for side in ['left','right']]
  env.close()
  for proc in robots:
   if proc is not None and proc.is_alive():proc.terminate();proc.join(timeout=5)
if __name__=='__main__':main()
