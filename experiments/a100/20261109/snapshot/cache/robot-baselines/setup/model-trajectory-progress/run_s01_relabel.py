"""Replay and verify S01 geometry, then publish each checked trajectory."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
from publish_s01_progress import BASE,OUT,publish,write
SETS=['ProPriMem-VLA-Failures-20261104','ProPriMem-IMPRINT-Extra-20261105']
rows=[(ds,json.loads(l)) for ds in SETS for l in (BASE/'viewer-datasets'/ds/'metadata.jsonl').read_text().splitlines() if json.loads(l)['task_id']=='S01']
def one(item):
 index,(ds,row)=item;key=row['sample_id'];log=OUT/ds/'logs'/(key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
 geometry=OUT/ds/'geometry'/(key+'.json')
 if not geometry.exists():
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(1+index%2),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
  with log.open('w') as f:
   subprocess.run([str(BASE/'envs/rmbench/bin/python'),str(Path(__file__).with_name('replay_s01_geometry.py')),ds,key],env=env,stdout=f,stderr=f,check=True)
 return publish(ds,key)
def main():
 done=[];errors=[];start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures={pool.submit(one,x):x[1] for x in enumerate(rows)}
  for future in concurrent.futures.as_completed(futures):
   ds,row=futures[future]
   try:done.append(future.result())
   except Exception as exc:errors.append(dict(dataset=ds,sample=row['sample_id'],error=repr(exc)))
   status=dict(status='running',target=len(rows),published=len(done),errors=errors,seconds=time.time()-start,time=time.time())
   write(OUT/'status.json',status);print(json.dumps(status),flush=True)
 status['status']='complete' if not errors else 'partial_needs_review';write(OUT/'status.json',status)
if __name__=='__main__':main()
