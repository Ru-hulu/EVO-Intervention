from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import av,hashlib,pickle,json,time,os,re
R=Path(__file__).parent;FULL=R/'full'
manifest=json.loads((FULL/'manifest.json').read_text())
TASKS={t['task_id']:t for t in manifest['tasks'][31:]}
EXPECTED={}
for r in json.loads((FULL/'provenance.json').read_text())['trajectories']:
 tid=r['sample_id'].split('_')[1];ep=int(r['sample_id'].rsplit('_',1)[1])
 rel=f"{TASKS[tid]['task_path']}/videos/{r['view']}/episode_{ep:06d}.mp4"
 EXPECTED[rel]=(r['decoded_yuv_sha256'],r['frames'])
def compact(path):
 p=Path(path);tmp=p.with_suffix('.compact.mp4');source_hash=hashlib.sha256();count=0
 rel=str(p.relative_to(FULL));expected,frames=EXPECTED[rel]
 with p.open('rb') as f:head=f.read(65536)
 if b'cabac=1' in head and b'subme=7' in head:
  h=hashlib.sha256();n=0
  with av.open(str(p)) as inp:
   rate=inp.streams.video[0].average_rate
   for f in inp.decode(video=0):
    assert abs(float(f.pts*f.time_base)-n/float(rate))<1e-4
    h.update(f.to_ndarray(format='yuv420p').tobytes());n+=1
  assert n==frames and h.hexdigest()==expected,rel
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  return rel,dict(video_sha256=digest,video_size_bytes=p.stat().st_size,decoded_yuv_sha256=h.hexdigest(),resumed=True)
 with av.open(str(p)) as inp,av.open(str(tmp),'w') as out:
  v=inp.streams.video[0];rate=v.average_rate
  s=out.add_stream('libx264',rate=rate,options={'crf':'0','preset':'medium'})
  s.width,s.height,s.pix_fmt=v.width,v.height,'yuv420p';s.codec_context.thread_count=2
  for i,f in enumerate(inp.decode(video=0)):
   source_hash.update(f.to_ndarray(format='yuv420p').tobytes());f.pts,f.time_base=i,1/rate
   for packet in s.encode(f):out.mux(packet)
   count+=1
  for packet in s.encode():out.mux(packet)
 h=hashlib.sha256();n=0
 with av.open(str(tmp)) as inp:
  for f in inp.decode(video=0):
   assert abs(float(f.pts*f.time_base)-n/float(rate))<1e-4
   h.update(f.to_ndarray(format='yuv420p').tobytes());n+=1
 assert n==count==frames and h.digest()==source_hash.digest() and h.hexdigest()==expected
 old=p.stat().st_size
 if tmp.stat().st_size<old:tmp.replace(p)
 else:tmp.unlink()
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 return str(p.relative_to(FULL)),dict(video_sha256=digest,video_size_bytes=p.stat().st_size,decoded_yuv_sha256=h.hexdigest(),original_size_bytes=old)
if __name__=='__main__':
 uploaded=set()
 for checkpoint in R.glob('uploaded-O*.pkl'):
  uploaded.update(op.path_in_repo for op in pickle.loads(checkpoint.read_bytes()))
 # Preserve the 48 objects confirmed by the earlier sequential uploader too.
 old=R/'preupload_progress.json'
 if old.exists():
  completed=json.loads(old.read_text())['videos']
  names=['main_view','left_view','right_view']
  for i in range(completed):uploaded.add(f'Order/08-enter_pin_sequence/videos/{names[i%3]}/episode_{i//3:06d}.mp4')
 files=[p for p in sorted((FULL/'Order').rglob('*.mp4')) if re.fullmatch(r'episode_\d{6}\.mp4',p.name) and p.relative_to(FULL).as_posix() not in uploaded]
 print('COMPACTING',len(files),'preserving',len(uploaded),'confirmed uploads',flush=True)
 results={}
 for partial in (FULL/'Order').rglob('episode_*.compact.mp4'):partial.unlink()
 with ProcessPoolExecutor(max_workers=3) as pool:
  for rel,details in pool.map(compact,files):
   results[rel]=details
   if len(results)%20==0:print('COMPACTED',len(results),'/',len(files),flush=True)
 (R/'compaction-results.json').write_text(json.dumps(results,ensure_ascii=False)+'\n')
 m=json.loads((FULL/'manifest.json').read_text())
 for task in m['tasks'][31:]:
  idx=FULL/task['trajectory_index'];rows=[json.loads(x) for x in idx.read_text().splitlines()]
  for row in rows:
   for view in row['video_views'].values():
    d=results.get(task['task_path']+'/'+view['video_path'])
    if d:view.update({k:d[k] for k in ['video_sha256','video_size_bytes']})
   row.update({k:row['video_views']['main_view'][k] for k in ['video_sha256','video_size_bytes']})
  idx.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
  task['trajectory_index_sha256']=hashlib.sha256(idx.read_bytes()).hexdigest()
 (FULL/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 for loc in [FULL/'provenance.json',FULL/'provenance/evomind-order-20261031/provenance.json']:
  prov=json.loads(loc.read_text())
  for r in prov['trajectories']:
   tid=r['sample_id'].split('_')[1]
   task=next(t for t in m['tasks'][31:] if t['task_id']==tid)
   ep=int(r['sample_id'].rsplit('_',1)[1]);rel=f"{task['task_path']}/videos/{r['view']}/episode_{ep:06d}.mp4"
   if rel in results:r.update({k:results[rel][k] for k in ['video_sha256','video_size_bytes','decoded_yuv_sha256']})
  prov['lossless_compaction']=dict(preset='medium',crf=0,pixels_verified=True,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  loc.write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n')
 import shutil
 shutil.copy2(__file__,FULL/'provenance/evomind-order-20261031/compact_lossless.py')
 receipt=json.loads((R/'pr-preparation.json').read_text())
 allfiles=[p for p in FULL.rglob('*') if p.is_file() and (p.relative_to(FULL).parts[0] in ['Order','provenance'] or p.name in ['manifest.json','README.md'])]
 before=receipt['bytes'];receipt.update(files=len(allfiles),bytes=sum(p.stat().st_size for p in allfiles))
 (R/'pr-preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
 # Revalidate all resulting files before the uploader is allowed to publish.
 (R/'publish-ready.json').unlink(missing_ok=True)
 print('COMPACTION DONE',before,'->',receipt['bytes'],flush=True)
