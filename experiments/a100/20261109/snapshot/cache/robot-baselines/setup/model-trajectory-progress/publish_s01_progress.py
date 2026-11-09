"""Publish only S01 labels, preserving prior labels and success metrics."""
import fcntl,json,shutil,time
from pathlib import Path
from s01_progress import curve,NEAR,RELEASE,LIFT,SHIFT
BASE=Path('/cache/robot-baselines');OUT=BASE/'annotations/s01-wrong-lid-20261105'
def read(p):return json.loads(p.read_text())
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False));tmp.replace(p)
def publish(ds,key):
 root=BASE/'viewer-datasets'/ds;meta=root/'metadata.jsonl'
 with (root/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  rows=[json.loads(l) for l in meta.read_text().splitlines()];row=next(r for r in rows if r['sample_id']==key);assert row['task_id']=='S01'
  backup=OUT/ds/'original';backup.mkdir(parents=True,exist_ok=True)
  pp=root/'progress'/(key+'.json');oldann=root/row['annotation_path']
  if not (backup/(key+'.progress.json')).exists():shutil.copy2(pp,backup/(key+'.progress.json'))
  if not (backup/(key+'.annotation.json')).exists():shutil.copy2(oldann,backup/(key+'.annotation.json'))
  if not (backup/(key+'.metadata.json')).exists():write(backup/(key+'.metadata.json'),row)
  old=read(backup/(key+'.progress.json'));g=read(OUT/ds/'geometry'/(key+'.json'))
  delta_root=BASE/'annotations'/('imprint-extra-20261105' if ds=='ProPriMem-IMPRINT-Extra-20261105' else 'boundary-integral-20261105')
  delta=read(delta_root/ds/'deltas'/(key+'.json'))
  p,freeze,loss,events=curve(delta['deltas'],old['weights'],g)
  assert len(p)==row['total_frames'] and g['max_qpos_error']<1e-4
  progress=dict(old,policy='s01_boundary_integral_v5_wrong_target_freeze_physical_rollback',dense_progress=p,events=events,first_issue=next((e for e in events if e['kind']=='regression'),None),source='成功参考共享权重；拿错盖子时冻结积分；实际破坏覆盖成果才扣对应贡献，修复后恢复；不按 fail_flag 清零',boundary_source='saved-action geometry replay; joint state and native phase/failure verified on every recorded frame',geometry_max_qpos_error=g['max_qpos_error'],geometry_source=str(OUT/ds/'geometry'/(key+'.json')),freeze_frames=[f for f,x in enumerate(freeze) if x],physical_rule=dict(near_m=NEAR,release_m=RELEASE,lift_m=LIFT,lateral_m=SHIFT,confirmation_frames=3),points=[[f,g['frames'][f]['phase']-len(loss[f])] for f in range(len(p)) if f==0 or g['frames'][f]['phase']!=g['frames'][f-1]['phase'] or loss[f]!=loss[f-1]],training_eligible=False,preview_only=True)
  ann=read(backup/(key+'.annotation.json'))
  boundaries=[0]+[f for f in range(1,len(p)) if g['frames'][f]['phase']!=g['frames'][f-1]['phase'] or loss[f]!=loss[f-1] or freeze[f]!=freeze[f-1]]
  stages=[]
  for i,f in enumerate(boundaries):
   valid=[f'phase:{j}' for j in range(1,g['frames'][f]['phase']+1) if f'phase:{j}' not in loss[f]]
   description='有效子任务：'+', '.join(valid)
   if freeze[f]:description+='；拿错盖子，progress 冻结'
   if loss[f]:description+='；已破坏：'+', '.join(sorted(loss[f]))
   stages.append(dict(stage_id=i,start_frame=f,end_frame=boundaries[i+1] if i+1<len(boundaries) else len(p),state_description=description))
  ann.update(progress=progress,stages=stages,num_stages=len(stages),annotation_status='s01_physical_progress_auto_needs_review',training_eligible=False)
  path=Path(key)/'annotation.s01-v5.json';write(root/path,ann);write(pp,progress)
  row.update(annotation_path=str(path),progress_policy=progress['policy'],annotation_status=ann['annotation_status'],training_eligible=False)
  tmp=meta.with_suffix('.s01.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(meta)
  result=dict(dataset=ds,sample=key,freeze_frames=sum(freeze),damage_events=[e for e in events if e['kind']=='regression'],old_final=old['dense_progress'][-1],new_final=p[-1],success=row['task_success'],time=time.time())
  write(OUT/ds/'reports'/(key+'.json'),result)
  return result
if __name__=='__main__':
 import sys
 print(json.dumps(publish(*sys.argv[1:3]),ensure_ascii=False))
