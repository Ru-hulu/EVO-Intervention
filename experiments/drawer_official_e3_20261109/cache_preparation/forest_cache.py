"""Publish a verified, unaugmented Forest drawer cache using existing causal base features."""
from pathlib import Path
import csv,fcntl,hashlib,io,json,os,shutil,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import torch
from PIL import Image
torch.set_num_threads(1)
S=Path('/cache/cvpzhr/setup/forest-drawer-noaug-cache-20261109')
D=Path('/cache/cvpzhr/datasets/forest-drawer-5fps-no-augmentation-20261109')
OLD=Path('/cache/cvpzhr/datasets/forest-five-anomaly-5fps-static-jitter-seed42-20261109')
RAW=Path('/cache/cvpzhr/datasets/failure_recover/drawer-inspection-close')
CODE=Path('/cache/cvpzhr/setup/forest-robometer-spray-official-e3-20261109/forest/code')
OLD_CODE=Path('/cache/cvpzhr/setup/forest-five-anomaly-20261109/code')
TASK='drawer-inspection-close'
sys.path.insert(0,str(CODE))
from mem_lite.datasets.m3_dataset import M3Dataset,VIEW_ORDER
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def expected_ids(frames):
 ids=list(range(0,frames,6))
 if ids[-1]!=frames-1:ids.append(frames-1)
 return ids
def link(src,dest):
 assert src.is_file() and not src.is_symlink(),src
 dest.parent.mkdir(parents=True,exist_ok=True);os.link(src,dest)

def worker(record,history_source,reference):
 r=dict(record);src=Path(r['base_dir']);target=D/r['split']/TASK/r['video_id'];part=target.with_name(target.name+'.partial')
 assert not target.exists() and not part.exists();part.mkdir(parents=True)
 summary=read(src/'summary.json');rows=[json.loads(l) for l in (src/'samples.jsonl').read_text().splitlines()]
 assert summary['history_source']==history_source and summary['history_capacity']==1008 and summary['history_dim']==2560
 assert summary['episode_id']==r['episode_id'] and summary['task_prompt']==r['task_prompt']
 assert [x['frame_index'] for x in rows]==expected_ids(r['frames']) and len(rows)==r['source_samples']==summary['num_samples']
 legacy_csv=Path(r['progress_csv']);assert sha(legacy_csv)==r['progress_sha256']
 old_labels=list(csv.DictReader(legacy_csv.open()))
 progress=RAW/'videos/main_view/annotation'/f"{r['video_id']}.progress.json"
 annotation=progress.with_name(f"{r['video_id']}.json");p=read(progress);ann=read(annotation)
 assert p['episode']==r['video_id'] and p['total_frames']==r['frames'] and len(p['frames'])==r['frames']
 assert [f['frame_index'] for f in p['frames']]==list(range(r['frames']))
 assert sha(annotation)==p['stage_annotation_sha256']
 assert sha(r['videos'][0])==p['video_sha256']==ann['video_sha256']
 assert len(old_labels)==r['frames']
 assert all(float(a['pt'])==float(b['pt']) and float(a['delta'])==float(b['delta']) for a,b in zip(p['frames'],old_labels))
 video_metadata=[]
 for video in r['videos']:
  meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,avg_frame_rate','-of','json',video],text=True))['streams'][0]
  assert int(meta['nb_frames'])==r['frames']
  a,b=map(int,meta['avg_frame_rate'].split('/'));assert a/b==r['fps']==30
  video_metadata.append({'path':video,**meta})
 labels=D/'labels'/TASK/legacy_csv.name;labels.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(legacy_csv,labels)
 r.update(events=[],added_samples=0,augmented_samples=r['source_samples'],augmentation_enabled=False,
          source_progress=str(progress),source_progress_sha256=sha(progress),source_annotation=str(annotation),source_annotation_sha256=sha(annotation),
          progress_csv=str(labels),progress_sha256=sha(labels),base_dir=str(target),output_dir=str(target),reused_from=str(src))
 checksums={};previous=None;histories=0;images=0;logical_bytes=0
 for i,row in enumerate(rows):
  key=(r['episode_id'],row['frame_index']);assert key in reference
  assert row['pt']==float(p['frames'][row['frame_index']]['pt'])==reference[key]['pt']
  assert reference[key]['kind']=='original' and reference[key]['task']==r['task_prompt']
  assert 0<=row['pt']<=1 and set(row['images'])==set(VIEW_ORDER)
  assert (row['history'] is None)==(i==0)
  for rel in row['images'].values():
   relpath=Path(rel);assert not relpath.is_absolute() and '..' not in relpath.parts
   raw=(src/rel).read_bytes()
   with Image.open(io.BytesIO(raw)) as im:assert im.width>0 and im.height>0;im.verify()
   checksums[rel]=hashlib.sha256(raw).hexdigest();logical_bytes+=len(raw);link(src/rel,part/rel);images+=1
  if row['history']:
   rel=row['history'];rp=Path(rel);assert not rp.is_absolute() and '..' not in rp.parts
   raw=(src/rel).read_bytes();h=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);tokens=h['tokens']
   assert h['through_frame']==previous and tokens.ndim==2 and tokens.shape[1]==2560 and 0<len(tokens)<=1008
   assert tokens.dtype==torch.bfloat16 and torch.isfinite(tokens).all().item()
   checksums[rel]=hashlib.sha256(raw).hexdigest();logical_bytes+=len(raw);link(src/rel,part/rel);histories+=1
  previous=row['frame_index']
 assert histories==len(rows)-1==len(list((src/'history').glob('*.pt')))
 assert images==3*len(rows)
 shutil.copyfile(src/'samples.jsonl',part/'samples.jsonl')
 summary.update(source=r,augmentation={'enabled':False,'inserted_observations':0,'stationary':False,'jitter':False},
                cache_provenance={'method':'verified reuse of unaugmented causal base cache','source_directory':str(src),'payload_storage':'hardlinked immutable PNG and history tensors'})
 save(part/'summary.json',summary);save(part/'payload-sha256.json',checksums)
 ds=M3Dataset([str(part)])
 for idx in [0,len(rows)-1]:
  sample=ds[idx];assert sample['target_pt']==rows[idx]['pt'] and len(sample['images'])==3
 del ds
 part.rename(target)
 receipt={'episode_id':r['episode_id'],'split':r['split'],'output_dir':str(target),'samples':len(rows),
          'all_histories_verified':histories,'all_images_verified':images,'all_three_views_present':True,
          'causal_history_through_previous_observation':True,'current_labels_equal_original_cached_labels':True,
          'labels_and_frames_equal_original_base_cache':True,'augmentation_enabled':False,'inserted_observations':0,
          'samples_sha256':sha(target/'samples.jsonl'),'summary_sha256':sha(target/'summary.json'),
          'payload_manifest_sha256':sha(target/'payload-sha256.json'),'logical_payload_bytes':logical_bytes,'video_metadata':video_metadata}
 save(D/'receipts'/f"{r['episode_id']}.json",receipt)
 return r,receipt

def main():
 S.mkdir(parents=True,exist_ok=True);lock=(S/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state={'pid':os.getpid(),'started_at':time.time(),'task':TASK,'gpu_used':False,'training_started':False,'automatic_training_start':False}
 def update(phase,**kw):state.update(phase=phase,updated_at=time.time(),**kw);save(S/'status.json',state)
 try:
  update('preflight');assert not D.exists(),'Inspect any pre-existing target before restarting'
  old_plan=read(OLD/'plan.json');records=[r for r in old_plan['records'] if r['task']==TASK];assert len(records)==27
  assert old_plan['history_source']==read(CODE.parent/'training.config.json')['history_source']
  code_files=[*sorted((OLD_CODE/'mem_lite/memory').glob('*.py')),OLD_CODE/'mem_lite/datasets/prepare_stream.py',OLD_CODE/'mem_lite/datasets/m3_dataset.py']
  code_proof={}
  for p in code_files:
   rel=p.relative_to(OLD_CODE);assert sha(p)==sha(CODE/rel);code_proof[str(rel)]=sha(p)
  reference={};split_keys={}
  for split in ['train','validation']:
   keys=set()
   for record in records:
    if record['split']!=split:continue
    source_rows=[json.loads(line) for line in (Path(record['base_dir'])/'samples.jsonl').read_text().splitlines()]
    for row in source_rows:
     key=(record['episode_id'],row['frame_index']);assert key not in reference
     reference[key]={'pt':row['pt'],'kind':'original','task':record['task_prompt']};keys.add(key)
   split_keys[split]=keys
  assert len(reference)==3857 and len(split_keys['train'])==3138 and len(split_keys['validation'])==719
  D.mkdir();done=[];history_count=image_count=logical_bytes=0
  update('validating_and_preparing',completed_episodes=0,total_episodes=27,cpu_workers=4)
  with ThreadPoolExecutor(max_workers=4) as pool:
   futures=[pool.submit(worker,r,old_plan['history_source'],reference) for r in records]
   for f in as_completed(futures):
    r,receipt=f.result();done.append(r);history_count+=receipt['all_histories_verified'];image_count+=receipt['all_images_verified'];logical_bytes+=receipt['logical_payload_bytes']
    update('validating_and_preparing',completed_episodes=len(done),completed_samples=sum(r['source_samples'] for r in done),verified_histories=history_count)
  done.sort(key=lambda r:r['episode_id'])
  plan={'format':'streamforest-clean-cache-v1','output_dir':str(D),'dataset_root':str(RAW),'model_path':old_plan['model_path'],
        'history_source':old_plan['history_source'],'history_dim':2560,'history_capacity':1008,'records':done,
        'augmentation_policy':{'enabled':False,'stationary':False,'jitter':False},
        'provenance':{'source_plan':str(OLD/'plan.json'),'source_plan_sha256':sha(OLD/'plan.json'),'code_equality_sha256':code_proof,'gpu_used':False}}
  plan['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest();save(D/'plan.json',plan)
  counts={};signatures={}
  for split in ['train','validation']:
   rr=[r for r in done if r['split']==split];paths=[r['output_dir'] for r in rr];keys=set()
   for r in rr:
    for l in (Path(r['output_dir'])/'samples.jsonl').read_text().splitlines():keys.add((r['episode_id'],json.loads(l)['frame_index']))
   assert keys==split_keys[split]
   ds=M3Dataset(paths,load_history=False);counts[split]={'episodes':len(rr),'samples':len(ds)};signatures[split]=ds.signature
   save(D/f'{split}.sample_files.json',paths);del ds
  assert counts=={'train':{'episodes':21,'samples':3138},'validation':{'episodes':6,'samples':719}}
  assert history_count==3830 and image_count==11571
  save(D/'split.json',{split:[r['episode_id'] for r in done if r['split']==split] for split in counts})
  report={'status':'completed','task':TASK,'counts':counts,'data_signatures':signatures,'plan_sha256':plan['plan_sha256'],
          'all_histories_verified':history_count,'all_images_verified':image_count,'all_three_views_present':True,
          'augmentation_enabled':False,'inserted_observations':0,'original_source_labels_preserved':True,
          'current_annotation_verified_pt_and_delta_identical':True,'labels_and_split_equal_original_dataset':True,
          'generation_mode':'verified reuse of unaugmented causal base caches; independent paths with immutable hardlinked payloads',
          'logical_payload_bytes':logical_bytes,'gpu_used':False,'training_started':False,'automatic_training_start':False,
          'elapsed_seconds':time.time()-state['started_at']}
  save(D/'cache-ready.json',report);save(S/'report.json',report);update('completed',completed_episodes=27,completed_samples=3857,result=report)
  print(json.dumps(report,ensure_ascii=False),flush=True)
 except BaseException as exc:
  update('failed',error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
