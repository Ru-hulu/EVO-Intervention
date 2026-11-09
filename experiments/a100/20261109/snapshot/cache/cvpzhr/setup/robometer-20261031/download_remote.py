from pathlib import Path
import concurrent.futures,hashlib,json,os,time,urllib.request,traceback
RUN=Path('/cache/cvpzhr/setup/robometer-20261031');MODEL=Path('/cache/cvpzhr/models/Robometer-4B');MODEL.mkdir(parents=True,exist_ok=True)
manifest=json.loads((RUN/'download-assets.json').read_text())
state={'phase':'downloading','pid':os.getpid(),'started':time.time(),'expected_bytes':sum(i['size'] for i in manifest)}
def save(**kw):
 state.update(kw);(RUN/'download-status.json').write_text(json.dumps(state,indent=2))
def fetch(i):
 target=MODEL/i['name'];target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists() and target.stat().st_size==i['size']: return i['name']
 part=target.with_suffix(target.suffix+'.part')
 for attempt in range(3):
  try:
   offset=part.stat().st_size if part.exists() else 0
   headers={'Range':f'bytes={offset}-'} if offset else {}
   with urllib.request.urlopen(urllib.request.Request(i['url'],headers=headers),timeout=60) as response:
    append=bool(offset and response.status==206)
    with part.open('ab' if append else 'wb') as f:
     while block:=response.read(8*1024**2):f.write(block)
   if part.stat().st_size!=i['size']:raise ValueError('size mismatch for '+i['name'])
   if i.get('sha256'):
    h=hashlib.sha256()
    with part.open('rb') as f:
     while b:=f.read(16*1024**2):h.update(b)
    if h.hexdigest()!=i['sha256']:raise ValueError('checksum mismatch for '+i['name'])
   part.replace(target);return i['name']
  except Exception as e:
   if attempt==2:raise RuntimeError(i['name']+': '+type(e).__name__)
   time.sleep(3)
try:
 save()
 with concurrent.futures.ThreadPoolExecutor(2) as pool:
  for name in pool.map(fetch,manifest):print('Verified',name,flush=True)
 save(phase='ready',sha256_verified=True,elapsed_seconds=time.time()-state['started'])
except Exception as e:save(phase='failed',error=str(e));traceback.print_exc();raise
