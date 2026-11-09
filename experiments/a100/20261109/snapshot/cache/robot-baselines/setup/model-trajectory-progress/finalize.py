"""Wait for complete evidence caches, integrate and publish reviewable labels.

Incomplete or unsupported trajectories retain their old annotations. Metadata
backups make the update reversible; success/failure outcomes are never changed.
"""
import json,time,shutil,hashlib,subprocess,sys
from pathlib import Path
from integrate import ROOT,OUT,DATASETS,read,write,main as integrate_all

def counts():
 expected={ds:len((ROOT/'viewer-datasets'/ds/'metadata.jsonl').read_text().splitlines()) for ds in DATASETS}
 deltas={ds:len(list((OUT/ds/'deltas').glob('*.json'))) for ds in DATASETS}
 replay={ds:len([p for p in (OUT/ds/'replay').glob('*.json') if not p.name.endswith('.error.json')]) for ds in DATASETS[1:]}
 return expected,deltas,replay

def publish():
 report=read(OUT/'integration-status.json');done=[]
 for ds in DATASETS:
  root=ROOT/'viewer-datasets'/ds;meta=root/'metadata.jsonl';backup=OUT/ds/'original';backup.mkdir(parents=True,exist_ok=True)
  if not (backup/'metadata.jsonl').exists():shutil.copy2(meta,backup/'metadata.jsonl')
  original=meta.read_bytes();rows=[json.loads(l) for l in original.decode().splitlines()]
  for row in rows:
   key=row['sample_id'];path=OUT/ds/'progress'/(key+'.json')
   if not path.exists():continue
   p=read(path);b=read(OUT/ds/'boundaries'/(key+'.json'));n=row['total_frames'];assert len(p['dense_progress'])==n
   oldpath=root/row['annotation_path'] if row.get('annotation_path') else None
   old=read(oldpath) if oldpath and oldpath.exists() else {}
   if old and not (backup/(key+'.json')).exists():write(backup/(key+'.json'),old)
   stages=[]
   for i,e in enumerate(b['events']):
    end=b['events'][i+1]['frame'] if i+1<len(b['events']) else n
    if end>e['frame']:stages.append(dict(stage_id=i,start_frame=e['frame'],end_frame=end,state_description='已完成子任务：'+', '.join(e['done'])))
   ann=dict(old,fps=row['fps'],total_frames=n,frame_index_base=0,frame_range_convention='[start_frame, end_frame)',stages=stages,num_stages=len(stages),progress=p,annotation_status='boundary_integral_auto_needs_review',training_eligible=False)
   relative=Path(key)/'annotation.integral-v4.json';write(root/relative,ann)
   progress_path=root/'progress'/(key+'.json')
   if progress_path.exists() and not (backup/(key+'.progress.json')).exists():shutil.copy2(progress_path,backup/(key+'.progress.json'))
   write(progress_path,p)
   row.update(annotation_path=str(relative),progress_policy=p['policy'],annotation_status=ann['annotation_status'],training_eligible=False)
   done.append(dict(dataset=ds,sample=key))
  assert meta.read_bytes()==original,'Metadata changed during publication; retry merge'
  tmp=meta.with_suffix('.integral.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(meta)
 write(OUT/'publication-status.json',dict(status='complete' if not report['pending'] else 'partial_needs_review',published=len(done),pending=report['pending'],issues=report['issues'],time=time.time(),training_eligible=False))

if __name__=='__main__':
 while True:
  expected,deltas,replay=counts();errors=list(OUT.glob('*/replay/*.error.json'))
  write(OUT/'pipeline-status.json',dict(expected=expected,deltas=deltas,replay=replay,replay_errors=[str(p) for p in errors],status='collecting_evidence',time=time.time()))
  if errors:raise RuntimeError('Replay verification failed; refusing publication')
  if deltas==expected and all(replay[ds]==expected[ds] for ds in DATASETS[1:]):break
  time.sleep(30)
 subprocess.run([sys.executable,str(Path(__file__).with_name('integrate.py'))],check=True);publish()
 print(json.dumps(read(OUT/'publication-status.json'),ensure_ascii=False),flush=True)
