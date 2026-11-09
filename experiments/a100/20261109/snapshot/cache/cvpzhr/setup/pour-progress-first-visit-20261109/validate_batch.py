"""Verify native-frame coverage, first-visit gating, PT recurrence, and source identity."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,math,datetime

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def validate(root,source):
 out=root/'progress_first_visit';main=root/'videos/main_view';expected={r['episode']:r for r in source['episodes']}
 assert len(expected)==28
 assert set(expected)=={p.stem for p in main.glob('episode_*.mp4')}=={p.stem for p in (main/'annotation').glob('episode_*.json')}
 assert set(expected)=={p.name.removesuffix('.progress.csv') for p in out.glob('episode_*.progress.csv')}
 assert set(expected)=={p.name.removesuffix('-delta.txt') for p in out.glob('episode_*-delta.txt')}
 assert set(expected)=={p.name.removesuffix('.metadata.json') for p in out.glob('episode_*.metadata.json')}
 assert not any(ep in expected for ep in ['episode_000002','episode_000004','episode_000024'])
 result=[];manifest=[]
 for ep,spec in sorted(expected.items()):
  ap=main/'annotation'/f'{ep}.json';vp=main/f'{ep}.mp4';cp=out/f'{ep}.progress.csv';dp=out/f'{ep}-delta.txt';mp=out/f'{ep}.metadata.json'
  ann=json.loads(ap.read_text());meta=json.loads(mp.read_text());n=ann['total_frames'];fps=ann['fps']
  assert digest(ap)==spec['annotation_sha256']==meta['annotation_sha256']
  assert digest(vp)==spec['video_sha256']==ann['video_sha256']==meta['video_sha256']
  hashes={p.name:digest(p) for p in [cp,dp,mp]}
  assert hashes[cp.name]==meta['progress_csv_sha256'] and hashes[dp.name]==meta['delta_sha256']
  deltas=[float(v) for v in dp.read_text().splitlines()]
  with cp.open() as f:rows=list(csv.DictReader(f))
  assert n==len(rows)==len(deltas)==spec['frames']==meta['total_frames']
  assert deltas[0]==0 and all(math.isfinite(d) and d>=0 for d in deltas)
  frame_stages=[None]*n;seen=Counter();first_segments=[];last_pour=None
  for s in ann['stages']:
   if s['action']=='pour':
    target=s['target_position'];seen[target]+=1;occurrence=seen[target];last_pour=(target,occurrence)
   else:
    assert s['action']=='put_down' and last_pour is not None
    target,occurrence=last_pour;assert s['after_pour_target']==target
   assert occurrence==s['target_occurrence']
   first=occurrence==1
   if first:first_segments.append(s['segment_index'])
   for i in range(s['source_start_frame'],s['source_end_frame']+1):
    assert frame_stages[i] is None
    frame_stages[i]=(s,target,occurrence,first)
  assert len(first_segments)==6 and set(seen)=={'left','middle','right'}
  weights={t:0. for t in seen};accepted=[];pts=[];modes=Counter();stage_sum=Counter()
  for i,(row,delta,stage) in enumerate(zip(rows,deltas,frame_stages)):
   assert int(row['frame_index'])==i and abs(float(row['time_seconds'])-i/fps)<1e-12
   assert float(row['delta'])==delta
   if stage:
    s,target,occ,first=stage;mode='first_visit' if first else 'repeat_hold'
    assert int(row['stage_id'])==s['stage_id'] and int(row['segment_index'])==s['segment_index']
    assert row['stage_name']==s['stage_name'] and row['target_position']==target and int(row['target_occurrence'])==occ
   else:
    first=False;mode='idle_hold';assert int(row['stage_id'])==int(row['segment_index'])==0
   assert row['mode']==mode and int(row['count_for_progress'])==int(first)
   credit=delta if first else 0.;assert float(row['credited_delta'])==credit
   accepted.append(credit);pts.append(float(row['pt']));modes[mode]+=1
   if stage:stage_sum[s['segment_index']]+=delta
   if first:weights[target]+=credit
  denominator=math.fsum(accepted)
  assert denominator>0 and abs(denominator-meta['normalization_denominator'])<1e-9
  assert pts[0]==0 and pts[-1]==1 and all(math.isfinite(p) and 0<=p<=1 for p in pts)
  for i in range(1,n):
   assert pts[i]>=pts[i-1]
   assert abs((pts[i]-pts[i-1])-accepted[i]/denominator)<1e-12
   if not rows[i]['count_for_progress']=='1':assert pts[i]==pts[i-1]
  for t,w in weights.items():assert abs(w/denominator-meta['cup_weights'][t])<1e-12
  complete=max(i for i,row in enumerate(rows) if row['count_for_progress']=='1')
  assert complete==meta['completion_frame'] and all(p==1 for p in pts[complete:])
  assert all(stage_sum[s]>0 for s in first_segments)
  if ep=='episode_000001':
   assert hashes[cp.name]==source['pilot_csv_sha256'] and hashes[dp.name]==source['pilot_delta_sha256']
  result.append({'episode':ep,'frames':n,'first_visit_frames':modes['first_visit'],'repeat_hold_frames':modes['repeat_hold'],'idle_hold_frames':modes['idle_hold'],'first_visit_action_segments':len(first_segments),'cup_weights':meta['cup_weights'],'completion_time_seconds':complete/fps,'checks_passed':True})
  manifest.append({'episode':ep,'frames':n,'files':hashes,'video_sha256':spec['video_sha256'],'annotation_sha256':spec['annotation_sha256']})
 report={'status':'passed','schema_version':'repeated-pouring-first-visit-progress-v1','rule_approved_date':source['rule_approved_date'],'validated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'episode_count':len(result),'total_frames':sum(r['frames'] for r in result),'new_episodes':27,'reused_pilot':'episode_000001','pilot_csv_and_raw_delta_unchanged':True,'first_visit_frames':sum(r['first_visit_frames'] for r in result),'repeat_hold_frames':sum(r['repeat_hold_frames'] for r in result),'idle_hold_frames':sum(r['idle_hold_frames'] for r in result),'all_frames_have_raw_delta_stage_and_pt':True,'repeat_and_idle_hold_exactly':True,'all_pt_monotonic_from_zero_to_one':True,'source_videos_and_annotations_unchanged':True,'episodes':result}
 index={'schema_version':'repeated-pouring-first-visit-progress-v1','rule_approved_date':source['rule_approved_date'],'pilot_user_approved':True,'episode_count':len(manifest),'total_frames':report['total_frames'],'native_fps':30,'stage_boundary_annotation_fps':5,'progress_rule':source['progress_rule'],'columns':list(rows[0]),'episodes':manifest}
 assert report['total_frames']==source['total_frames']==45395
 return report,index

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--source-manifest',type=Path,required=True);p.add_argument('--write',action='store_true');a=p.parse_args()
 report,index=validate(a.dataset_root,json.loads(a.source_manifest.read_text()))
 if a.write:
  for name,obj in [('batch_validation.json',report),('manifest.json',index)]:
   target=a.dataset_root/'progress_first_visit'/name;tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(target)
 print(json.dumps({k:v for k,v in report.items() if k!='episodes'},ensure_ascii=False,indent=2))
