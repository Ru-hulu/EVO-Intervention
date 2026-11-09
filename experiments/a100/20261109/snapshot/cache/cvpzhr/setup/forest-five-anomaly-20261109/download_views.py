from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,hashlib,urllib.request,urllib.error,time,os,fcntl
S=Path(__file__).resolve().parent
M=json.loads((S/'hf_snapshot/download_manifest.json').read_text());ROOT=Path(M['destination_root'])
def save(name,d):
 p=S/name;tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def get(row):
 rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],f'Existing file mismatch: {rel}'
  return row['path'],row['bytes']
 part=p.with_suffix('.mp4.part')
 for attempt in range(4):
  try:
   h=hashlib.sha256();n=0
   with urllib.request.urlopen(urllib.request.Request(row['download_url'],headers={'User-Agent':'Forest-data-preparation/1.0'}),timeout=60) as res,part.open('wb') as f:
    for block in iter(lambda:res.read(1024*1024),b''):f.write(block);h.update(block);n+=len(block)
   assert n==row['bytes'] and h.hexdigest()==row['sha256'],f'Download checksum mismatch: {rel}'
   part.replace(p);return row['path'],n
  except Exception as e:
   if attempt==3:raise RuntimeError(type(e).__name__+': '+str(e).split('?')[0][:200]) from None
   time.sleep(2**attempt)
if __name__=='__main__':
 lock=(S/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state={'phase':'downloading','pid':os.getpid(),'total_files':len(M['files']),'total_bytes':M['bytes_to_download'],'completed_files':0,'completed_bytes':0,'errors':[],'remote_direct_download':True,'started_at':time.time()};save('download-status.json',state)
 with ThreadPoolExecutor(6) as pool:
  pending={pool.submit(get,row):row for row in M['files']}
  for f in as_completed(pending):
   row=pending[f]
   try:
    name,size=f.result();state['completed_files']+=1;state['completed_bytes']+=size
   except Exception as e:state['errors'].append({'path':row['path'],'error':str(e)})
   state['updated_at']=time.time();save('download-status.json',state)
 state.update(phase='failed' if state['errors'] else 'completed',finished_at=time.time());save('download-status.json',state)
 print(json.dumps(state,ensure_ascii=False),flush=True)
 if state['errors']:raise SystemExit(1)
