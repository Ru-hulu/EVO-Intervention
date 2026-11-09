"""Re-label IMPRINT cover trajectories with the supplied fixed task rule."""
import json,hashlib,math,shutil,fcntl,time,sys,os
from pathlib import Path
from s01_fixed_rule import build
B=Path('/cache/robot-baselines');S=B/'setup/model-trajectory-progress';O=Path(os.environ.get('COVER_ANNOTATION_ROOT',str(B/'annotations/imprint-cover-fixed-20261105')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False));t.replace(p)
def main():
 rulep=S/'cover_blocks_progress.json';rule=json.loads(rulep.read_text());wp=S/rule['task_weights']['path'];assert sha(wp)==rule['task_weights']['weights_file_sha256'];weights=rule['task_weights']['weights'];assert json.loads(wp.read_text())['weights']==weights
 prepared=[];errors=[]
 selected=set(json.loads(Path(os.environ['COVER_SELECTED_IDS']).read_text())) if os.environ.get('COVER_SELECTED_IDS') else None
 for ds in json.loads(os.environ.get('COVER_DATASETS','["ProPriMem-VLA-Failures-20261104","ProPriMem-IMPRINT-Extra-20261105"]')):
  root=B/'viewer-datasets'/ds
  for line in (root/'metadata.jsonl').read_text().splitlines():
   row=json.loads(line);key=row['sample_id']
   if row['task_id']!='S01' or not row['model_name'].startswith('IMPRINT') or selected is not None and key not in selected:continue
   try:
    evidence=Path(row['evidence_path']);states=[json.loads(l) for l in (evidence/'states.jsonl').read_text().splitlines()]
    gp=B/'annotations/s01-wrong-lid-20261105'/ds/'geometry'/(key+'.json');geometry=json.loads(gp.read_text())
    dp=Path(os.environ.get('COVER_DELTA_ROOT',str(B/'annotations'/('imprint-extra-20261105' if 'Extra' in ds else 'boundary-integral-20261105'))))/ds/'deltas'/(key+'.json');d=json.loads(dp.read_text());video=root/row['video_path'];assert sha(video)==d['video_sha256']
    actions=[json.loads(l) for l in (evidence/'actions.jsonl').read_text().splitlines()];byframe={a['before_frame']:a['args'][0] for a in actions}
    controls=[];last=actions[0]['args'][0]
    for f in range(len(states)):
     if f in byframe:last=byframe[f]
     controls.append(last)
    result=build(d['deltas'],geometry,states,controls,weights)
    boundary=dict(schema='imprint_cover_fixed_boundaries_v1',sample_id=key,frame_ranges='zero based [a,b)',normal_stages=result['normal_stages'],wrong_frames=result['wrong_frames'],stationary_frames=result['stationary_frames'],failure_frame=result['failure_frame'],source_states_sha256=sha(evidence/'states.jsonl'),source_actions_sha256=sha(evidence/'actions.jsonl'),geometry_sha256=sha(gp),method='automatic geometry and control-target motion boundaries; review required')
    first_lift=None;first_uncovered=None;wrong_target=None
    for f,frame in enumerate(geometry['frames']):
     phase=frame['phase']
     if not 3<=phase<6:continue
     expected=geometry['open_order'][phase-3]
     for lid,(cover,block) in enumerate(zip(frame['covers'],frame['blocks'])):
      if lid==expected or lid in geometry['open_order'][:phase-3]:continue
      covered=abs(cover[0]-block[0])<.03 and abs(cover[1]-block[1])<.03 and cover[2]<.742
      if cover[2]>.743 and first_lift is None:first_lift=f;wrong_target=lid
      if not covered and first_uncovered is None:first_uncovered=f
    boundary.update(initial_color_to_cover=geometry['open_order'],first_wrong_lift_frame=first_lift,first_wrong_uncovered_frame=first_uncovered,wrong_cover=wrong_target,native_failure_frame=result['failure_frame'])
    bp=O/ds/'boundaries'/(key+'.json');dump(bp,boundary)
    progress_path = root/'progress'/(key+'.json')
    if progress_path.exists():
     old = json.loads(progress_path.read_text())
    else:
     old = json.loads((root/row['annotation_path']).read_text()).get('progress', {})
    p=dict(old,total=6,policy='imprint_cover_fixed_task_weights_native_failure_v6',weights=weights,dense_progress=result['dense_progress'],freeze_frames=result['wrong_frames'],frozen_from=result['failure_frame'],recoverable=False if result['failure_frame'] is not None else True,real_progress_regression=result['failure_frame'] is not None,normalization='fixed clean-demo task weights; never renormalized per trajectory',rule_sha256=sha(rulep),weights_file_sha256=sha(wp),boundary_json_sha256=sha(bp),delta_sha256=sha(dp),video_sha256=sha(video),boundary_source=boundary['method'],source='Memory Recover 固定六步权重；正常接近与必要撤手按原视频视觉 delta 积分；错误动作和静止平台；首次原生 fail_flag 起永久归零',preview_only=True,training_eligible=False)
    for name in ['reference_samples','physical_rule','geometry_max_qpos_error','geometry_source','unresolved_intervals']:p.pop(name,None)
    events=[dict(frame=s['end_frame']-1,kind='complete',progress=p['dense_progress'][s['end_frame']-1],step=s['nowid']) for s in result['normal_stages']]
    for f in range(1,len(states)):
     if f in set(result['wrong_frames']) and f-1 not in set(result['wrong_frames']):events.append(dict(frame=f,kind='wrong_approach',progress=p['dense_progress'][f]))
    failure=result['failure_frame'];issue=dict(frame=failure,kind='irreversible_failure',progress=0) if failure is not None else None
    if issue:events.append(issue)
    p.update(events=sorted(events,key=lambda x:x['frame']),first_issue=issue,points=[[0,0]]+[[s['end_frame']-1,int(s['nowid'])] for s in result['normal_stages']]+([[failure,0]] if failure is not None else []))
    dump(O/ds/'progress'/(key+'.json'),p);prepared.append((ds,row,result))
   except Exception as exc:errors.append(dict(dataset=ds,sample=key,error=repr(exc)))
 dump(O/'status.json',dict(status='prepared',prepared=len(prepared),errors=errors,time=time.time()));print('PREPARED',len(prepared),'ERRORS',errors,flush=True)
 if '--publish' not in sys.argv or errors:return
 for ds in sorted({ds for ds,_,_ in prepared}):
  root=B/'viewer-datasets'/ds;meta=root/'metadata.jsonl'
  with (root/'.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);rows=[json.loads(l) for l in meta.read_text().splitlines()]
   for _,original,result in [v for v in prepared if v[0]==ds]:
    row=next(r for r in rows if r['sample_id']==original['sample_id']);key=row['sample_id'];backup=O/ds/'original';backup.mkdir(parents=True,exist_ok=True)
    for src,name in [(root/'progress'/(key+'.json'),key+'.progress.json'),(root/row['annotation_path'],key+'.annotation.json')]:
     if src.exists() and not (backup/name).exists():shutil.copy2(src,backup/name)
    dump(backup/(key+'.metadata.json'),row)
    p=json.loads((O/ds/'progress'/(key+'.json')).read_text());ann=json.loads((root/row['annotation_path']).read_text());n=row['total_frames'];normal=result['normal_stages'];wrong=set(result['wrong_frames']);stationary=set(result['stationary_frames']);fail=result['failure_frame']
    def label(f):
     if fail is not None and f>=fail:return '原生不可恢复失败，P 永久归零'
     if f in wrong:return '错误盖子接近／错误动作，P 保持平台'
     step=next((s['nowid'] for s in normal if s['start_frame']<=f<s['end_frame']),None)
     if step and f not in stationary:return rule['subtasks'][int(step)-1]['name']+'：正常操作／必要撤手'
     return '静止等待或未完成子任务，P 保持平台'
    starts=[0]+[f for f in range(1,n) if label(f)!=label(f-1)];stages=[dict(stage_id=i,start_frame=f,end_frame=starts[i+1] if i+1<len(starts) else n,state_description=label(f)) for i,f in enumerate(starts)]
    ann.update(progress=p,stages=stages,num_stages=len(stages),annotation_status='fixed_cover_rule_auto_needs_boundary_review',training_eligible=False)
    ap=Path(key)/'annotation.cover-fixed-v6.json';dump(root/ap,ann);dump(root/'progress'/(key+'.json'),p)
    row.update(annotation_path=str(ap),progress_policy=p['policy'],annotation_status=ann['annotation_status'],training_eligible=False)
   tmp=meta.with_suffix('.fixed.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(meta)
  for file in [rulep,wp]:
   dest=root/'code/rules'/file.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
 dump(O/'status.json',dict(status='published',prepared=len(prepared),errors=[],time=time.time()));print('PUBLISHED',len(prepared),flush=True)
if __name__=='__main__':main()
