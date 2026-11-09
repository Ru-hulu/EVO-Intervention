"""Conservative event-based progress; raw evidence retained for review."""
from pathlib import Path
import json,os,fcntl,time,sys
import numpy as np
R=Path('/cache/robot-baselines');OUT=R/'eval-results/imprint-extra-20261105';DS=R/'viewer-datasets/ProPriMem-IMPRINT-Extra-20261105'
TASKS=json.loads((R/'setup/proprimem-rmbench-20261104/plan.json').read_text(encoding='utf-8'))['tasks']
TASKS=[t for t in TASKS if t['sim_task']!='place_block_mat']
MODELS={'simplemem':'SimpleMemVLA','full_attention':'IMPRINT 完整注意力','recurrent':'IMPRINT 循环记忆'}
def inside(pose,target,eps=.03):return bool(np.max(np.abs(np.array(pose)[:2]-np.array(target)[:2]))<eps and pose[2]<.77)
def progress(task,states):
 dense=[];events=[];seen=set();ever_center=False
 for s in states:
  poses=s['poses'];ok=s['success'];desc={};v=0
  if task=='press_button':
   counts=[s[k] for k in ['press_cnt_1','press_cnt_2','press_cnt_check_button']];goals=[s['card_id_1'],s['card_id_2'],1]
   v=0 if any(a>b for a,b in zip(counts,goals)) else sum(min(a,b) for a,b in zip(counts,goals))/sum(goals);desc={'actual_counts':counts,'required_counts':goals}
  elif task=='cover_blocks':
   v=s['current_state_pointer']/(len(s['target_state_transition'])-1)
   if s['fail_flag']:v=0
   desc={'native_state':s['current_state_pointer'],'irreversible_failure':s['fail_flag']}
  elif task in ['put_back_block','rearrange_blocks']:
   if task=='put_back_block':at=inside(poses['block'][0]['p'],[.1,-.1])
   else:at=inside(poses['block1'][0]['p'],poses['final_occupy_mat'][0]['p'])
   ever_center=ever_center or at;v=(int(ever_center)+int(s['stage_id']>=1)+int(ok))/3;desc={'checkpoint_seen':ever_center,'native_stage':s['stage_id'],'press_count':s['press_cnt']}
  elif task=='place_block_mat':
   done=[x=='True' for x in s['flag']];home=[inside(b['p'],poses['mat_lst'][int(i)]['p']) for b,i in zip(poses['blocks_lst'],s['selected_idx'])];v=sum(a and b for a,b in zip(done,home))/3;desc={'visited_green':done,'currently_home':home}
  elif task=='battery_try':
   bs=[poses[k][0] for k in ['battery1','battery2']];targets=[s['target_pose1_p'],s['target_pose2_p']]
   if all(inside(b['p'],t) for b,t in zip(bs,targets)):
    combo=tuple(int(abs(np.dot(b['q'],[0,0,.707,-.707]))>abs(np.dot(b['q'],[.707,-.707,0,0]))) for b in bs);seen.add(combo)
   v=min(.75,len(seen)/4);desc={'unique_inserted_combinations':sorted(seen),'policy':'search_coverage_lower_bound'}
  elif task=='blocks_ranking_try':
   if s['press_cnt']>len(seen):
    bs=[poses[k][0]['p'] for k in ['block1','block2','block3']];key=tuple(sorted(range(3),key=lambda i:bs[i][0]));seen.add(key)
   v=min(5/6,len(seen)/6);desc={'unique_checked_permutations':sorted(seen),'policy':'search_coverage_lower_bound'}
  if ok:v=1
  v=float(min(1,max(0,v)));frame=s['frame']
  if not dense or abs(v-dense[-1])>1e-8:events.append(dict(frame=frame,kind='complete' if not dense or v>dense[-1] else 'regression',progress=v,evidence=desc))
  dense.append(v)
 return dense,events

def publish():
 # Apply task removal to the resident worker after its current complete episode.
 main=sys.modules.get('__main__')
 if getattr(main,'MODEL',None) in ('full_attention','recurrent') and hasattr(main,'TASKS'):
  main.TASKS[:]=[t for t in TASKS if t['sim_task'] in ('cover_blocks','blocks_ranking_try')]
 DS.mkdir(parents=True,exist_ok=True)
 with (DS/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);rows=[]
  for status in sorted(OUT.glob('*/*/seed_*/status.json')):
   x=json.loads(status.read_text(encoding='utf-8'))
   if x.get('status')!='complete' or x.get('task')=='place_block_mat':continue
   task=next(t for t in TASKS if t['sim_task']==x['task']);e=Path(x['evidence']);meta=json.loads((e/'evidence.json').read_text(encoding='utf-8'));states=[json.loads(l) for l in (e/'states.jsonl').read_text(encoding='utf-8').splitlines()];n=meta['frames']
   assert n==len(states)>0
   key=f"{task['task_id']}-{x['model']}-seed{x.get('trial_id', x['seed'])}";folder=DS/key;folder.mkdir(exist_ok=True);views={}
   for view,cam in zip(['main_view','left_view','right_view'],['head_camera','left_camera','right_camera']):
    src=e/(cam+'.mp4');dest=folder/(view+'.mp4')
    if not dest.exists():os.link(src,dest)
    views[view]={'video_path':str(dest.relative_to(DS))}
   dense,events=progress(x['task'],states);stages=[]
   for i,event in enumerate(events):stages.append(dict(stage_id=i,start_frame=event['frame'],end_frame=events[i+1]['frame'] if i+1<len(events) else n,state_description=json.dumps(event['evidence'],ensure_ascii=False)))
   first=next((a for a in events if a['kind']=='regression'),None)
   policy='native-state-event-lower-bound-v1';review=x['task'] in ['battery_try','blocks_ranking_try','press_button']
   ann=dict(fps=meta['fps'],total_frames=n,frame_index_base=0,frame_range_convention='[start_frame, end_frame)',stages=stages,num_stages=len(stages),progress=dict(policy=policy,dense_progress=dense,events=events,first_issue=first,recoverable=None,frozen_from=None),annotation_status='awaiting_boundary_integral')
   (folder/'annotation.json').write_text(json.dumps(ann,ensure_ascii=False),encoding='utf-8');pd=DS/'progress';pd.mkdir(exist_ok=True);None  # New trajectories wait for boundary-integral progress
   rows.append(dict(sample_id=key,task_id=task['task_id'],task_name=task['task_id']+'｜'+task['task_name'],model_name=MODELS[x['model']],episode_index=meta['seed'],fps=meta['fps'],total_frames=n,video_path=views['main_view']['video_path'],video_views=views,annotation_path=key+'/annotation.json',task_prompt=task['source_instruction'],source_instruction=task['source_instruction'],task_success=meta['success'],failure_type=None if meta['success'] else 'native_task_incomplete',failure_reason=MODELS[x['model']]+'｜'+('成功对照' if meta['success'] else '原生任务未完成'),annotation_status=ann['annotation_status'],training_eligible=False,actual_seed=meta['seed'],renderer_spp=meta['renderer_spp'],evidence_path=str(e)))
  p=DS/'metadata.jsonl.tmp';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');p.replace(DS/'metadata.jsonl')
  counts={m:{t['sim_task']:sum(r['model_name']==MODELS[m] and r['task_id']==t['task_id'] and not r['task_success'] for r in rows) for t in TASKS} for m in MODELS}
  (DS/'collection_status.json').write_text(json.dumps(dict(target_episodes=80,failure_counts=counts,rows=len(rows)),indent=2))
 # Pause only after a complete episode has been exported; no trajectory is interrupted.
 migration=R/'setup/imprint-extra-20261105/pause-recurrent-for-migration'
 if os.environ.get('VLA_WORKER_MODEL')=='recurrent' and migration.exists():
  (migration.parent/'recurrent-paused.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
  while migration.exists():time.sleep(2)
 gate=R/'setup/imprint-extra-20261105/pause-full-for-allocation'
 if os.environ.get('VLA_WORKER_MODEL')=='full_attention' and gate.exists():
  (gate.parent/'full-paused.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time())))
  while gate.exists():time.sleep(2)
if __name__=='__main__':publish()
