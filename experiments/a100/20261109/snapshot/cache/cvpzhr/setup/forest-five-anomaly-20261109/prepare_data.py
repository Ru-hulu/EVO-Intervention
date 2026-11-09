"""CPU-only source checks and preparation for the original Forest protocol."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from fractions import Fraction
import csv,hashlib,json,math,time,sys
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
import prepare_static_jitter as aug
ROOT=Path('/cache/cvpzhr/datasets/failure_recover')
OUT=Path('/cache/cvpzhr/datasets/forest-five-anomaly-5fps-static-jitter-seed42-20261109')
PROMPTS={
'flip-two-cups-left-to-right-with-recovery':'Flip the two cups one at a time from left to right. Turn each cup over and then flip it back to its original orientation before moving on to the next cup.',
'pour-water-into-three-cups-with-repeated-pouring':'Pour water into each of three identical cups in any order.',
'drawer-inspection-close':'Open each of the three drawers in any order, look inside, and close it again.',
'zipper-pencil-case-inspection':'Inspect each of the three pencil cases. Pick it up, open and close its zipper, then put it back.',
'inspect-spray-bottles':'Pick up each of the three spray bottles from the rack, set it on the table, then return it to its original position in the rack.'}
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def indices(n):
 result=list(range(0,n,6))
 if result[-1]!=n-1:result.append(n-1)
 return result

def prepare():
 if (OUT/'plan.json').exists():
  print('Prepared plan already exists; not replacing it.');return
 hf=read(S/'hf_snapshot/hf-manifest.json');snapshot=read(S/'hf_snapshot/hf-files.json');files={r['path']:r for r in snapshot['files']};ref=read(S/'normal-reference-config.json')
 assert hf['totals']=={'tasks':5,'train':124,'val':33,'trajectories':157}
 records=[];video_jobs=[];sources={};started=time.monotonic()
 save(S/'preparation-status.json',{'phase':'checking_labels','training_started':False})
 for task in hf['tasks']:
  slug=task['task_path'];path=ROOT/slug;rows=[json.loads(l) for l in (S/'hf_snapshot'/slug/'metadata.jsonl').read_text().splitlines()]
  split={sid:kind for kind,key in [('train','train'),('validation','val')] for sid in task[key]}
  assert len(split)==len(rows) and set(split)=={r['sample_id'] for r in rows}
  eps={f"episode_{r['episode_index']:06}" for r in rows}
  assert eps=={p.stem for p in (path/'videos/main_view').glob('episode_*.mp4')}
  for row in rows:
   ep=f"episode_{row['episode_index']:06}";n=row['total_frames'];sid=f"{task['suite']}/{slug}/{ep}";kind=split[row['sample_id']]
   if slug=='flip-two-cups-left-to-right-with-recovery':
    pp=path/'progress_13stages'/f'{ep}.progress.csv';mp=path/'progress_13stages'/f'{ep}.metadata.json';ap=path/'videos/annotation'/f'{ep}.json'
   elif slug=='pour-water-into-three-cups-with-repeated-pouring':
    pp=path/'progress_first_visit'/f'{ep}.progress.csv';mp=path/'progress_first_visit'/f'{ep}.metadata.json';ap=path/'videos/main_view/annotation'/f'{ep}.json'
   else:
    pp=path/'videos/main_view/annotation'/f'{ep}.progress.json';mp=pp;ap=path/'videos/main_view/annotation'/f'{ep}.json'
   meta=read(mp);annotation=read(ap)
   if pp.suffix=='.csv':
    with pp.open() as f:labels=list(csv.DictReader(f))
    assert sha(pp)==meta['progress_csv_sha256'] and sha(ap)==meta['annotation_sha256']
   else:
    labels=meta['frames'];assert sha(ap)==meta['stage_annotation_sha256']
   assert n==len(labels)==meta['total_frames']==annotation['total_frames']
   assert [int(v['frame_index']) for v in labels]==list(range(n))
   pt=[float(v['pt']) for v in labels];delta=[float(v['delta']) for v in labels]
   assert all(math.isfinite(v) and 0<=v<=1 for v in pt) and pt[0]==0 and math.isclose(pt[-1],1,abs_tol=1e-12)
   assert delta[0]==0 and all(math.isfinite(v) and v>=0 for v in delta)
   # Intentional rollbacks remain present; no monotonicity filter or smoothing.
   rollbacks=sum(b<a-1e-12 for a,b in zip(pt,pt[1:]))
   videos=[]
   for view in ['main_view','left_view','right_view']:
    rel=f'{slug}/videos/{view}/{ep}.mp4';spec=files[rel];vp=ROOT/rel;videos.append(str(vp));video_jobs.append((vp,spec,n))
    if view=='main_view':assert spec['sha256']==meta['video_sha256']==annotation['video_sha256']
   for p in {pp,mp,ap}:sources[str(p)]=sha(p)
   normalized=OUT/'labels'/slug/f'{ep}.progress.csv';normalized.parent.mkdir(parents=True,exist_ok=True)
   with normalized.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['frame_index','delta','pt']);w.writerows(zip(range(n),delta,pt))
   sampled=indices(n);events=aug.choose_events([{'frame_index':i} for i in sampled],sid,Fraction(30),n) if kind=='train' else []
   added=sum(e['frames']*(2 if e['kind']=='jitter' else 1) for e in events)
   records.append({'episode_id':sid,'task':slug,'suite':task['suite'],'source_id':row['sample_id'],'video_id':ep,'split':kind,'fps':30,'frames':n,'task_prompt':PROMPTS[slug],'videos':videos,'progress_csv':str(normalized),'progress_sha256':sha(normalized),'source_progress':str(pp),'source_progress_sha256':sha(pp),'source_annotation':str(ap),'source_annotation_sha256':sha(ap),'rollbacks':rollbacks,'source_samples':len(sampled),'events':events,'added_samples':added,'augmented_samples':len(sampled)+added,'base_dir':str(OUT/'base'/kind/sid),'output_dir':str(OUT/'augmented/train'/sid) if kind=='train' else str(OUT/'base/validation'/sid)})
 assert len(records)==157 and sum(r['frames'] for r in records)==258438
 save(S/'preparation-status.json',{'phase':'waiting_for_views','labels_checked':157,'frames':258438,'training_started':False})
 while True:
  download=read(S/'download-status.json')
  if download['phase']=='failed':raise RuntimeError('View download failed; see download-status.json')
  if download['phase']=='completed':break
  time.sleep(10)
 save(S/'preparation-status.json',{'phase':'checking_all_three_views','videos':len(video_jobs),'training_started':False})
 def inspect(job):
  import av
  vp,spec,n=job
  assert vp.is_file() and vp.stat().st_size==spec['bytes'] and sha(vp)==spec['sha256'],str(vp)
  with av.open(str(vp)) as c:
   stream=c.streams.video[0];stream.codec_context.thread_count=1;assert float(stream.average_rate)==30
   count=0;first=None;last=None
   for frame in c.decode(stream):
    ts=float(frame.pts*stream.time_base);assert abs(ts-count/30)<1e-4,(str(vp),count,ts)
    if first is None:first=ts
    last=ts;count+=1
   assert count==n,(str(vp),count,n)
  return {'path':str(vp),'frames':count,'fps':30,'sha256':spec['sha256'],'bytes':spec['bytes']}
 with ThreadPoolExecutor(8) as pool:checked=list(pool.map(inspect,video_jobs))
 for v in checked:sources[v['path']]=v['sha256']
 config={k:v for k,v in ref.items() if k not in ['sample_files','validation_sample_files','validation_suite_by_episode','preparation_hashes','data_signature','validation_data_signature']}
 config.update(output_dir='/cache/cvpzhr/outputs/forest-five-anomaly-frombase-e3-20261109',sample_files=[r['output_dir'] for r in records if r['split']=='train'],validation_sample_files=[r['output_dir'] for r in records if r['split']=='validation'],validation_suite_by_episode={r['output_dir']:r['task'] for r in records if r['split']=='validation'})
 assert not config.get('init_checkpoint') and not config.get('resume')
 save(S/'training.config.prepared.json',config)
 counts={slug:{kind:sum(r['task']==slug and r['split']==kind for r in records) for kind in ['train','validation']} for slug in PROMPTS}
 report={'status':'passed','tasks':counts,'episode_count':157,'total_source_frames':258438,'three_view_videos':471,'three_view_frames_synchronized':True,'source_pt_preserved':True,'rollback_counts_by_task':{slug:sum(r['rollbacks'] for r in records if r['task']==slug) for slug in PROMPTS},'base_observations':{k:sum(r['source_samples'] for r in records if r['split']==k) for k in ['train','validation']},'final_observations':{k:sum(r['augmented_samples'] for r in records if r['split']==k) for k in ['train','validation']},'added_training_observations':sum(r['added_samples'] for r in records),'validation_augmented':False,'task_prompts':PROMPTS,'training_started':False,'seconds':time.monotonic()-started}
 plan={'format':'forest-five-anomaly-original-v1','dataset_root':str(ROOT),'output_dir':str(OUT),'model_path':ref['model_path'],'history_source':ref['history_source'],'history_dim':ref['model']['history_dim'],'history_capacity':ref['history_capacity'],'hf_revision':snapshot['revision'],'augmentation_policy':read(S/'augmentation-policy.json'),'sources':sources,'records':records,'code_sha256':read(S/'reference-code-sha256.json')}
 plan['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();save(OUT/'plan.json',plan);save(OUT/'source-preflight.json',report);save(S/'preparation-status.json',{'phase':'ready_for_gpu','training_started':False,**report})
 save(OUT/'split.json',{slug:{kind:[r['video_id'] for r in records if r['task']==slug and r['split']==kind] for kind in ['train','validation']} for slug in PROMPTS})
 print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
 try:prepare()
 except Exception as e:
  save(S/'preparation-status.json',{'phase':'failed','error':repr(e),'training_started':False});raise
