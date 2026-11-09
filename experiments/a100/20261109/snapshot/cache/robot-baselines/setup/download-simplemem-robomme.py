import concurrent.futures,hashlib,json,time,shutil
from pathlib import Path
import requests
BASE=Path('/cache/robot-baselines');TARGET=BASE/'models/SimpleMemVLA/RoboMME'
REPO='yinchenghust/simplememvla_robomme';REV='6bf7204311496b93f12889cad0eaf3fcbddc35e4';ENDPOINT='https://hf-mirror.com'
TARGET.mkdir(parents=True,exist_ok=True)
def session():
 s=requests.Session();s.trust_env=False;return s
s=session();r=s.get(f'{ENDPOINT}/api/models/{REPO}/revision/{REV}?blobs=true',timeout=45);r.raise_for_status();meta=r.json();assert meta['sha']==REV
files=[x for x in meta['siblings'] if x['rfilename']!='.DS_Store']
manifest={'repo_id':REPO,'revision':REV,'endpoint':ENDPOINT,'target':str(TARGET),'files':files}
(BASE/'manifests/simplemem-robomme-source.json').write_text(json.dumps(manifest,indent=2))
start=time.time();results=[]
for item in files:
 name=item['rfilename'];size=item['size'];path=TARGET/name;path.parent.mkdir(parents=True,exist_ok=True)
 url=f'{ENDPOINT}/{REPO}/resolve/{REV}/{name}'
 if not path.exists():
  if size<64*1024*1024:
   r=s.get(url,timeout=(30,120));r.raise_for_status();path.with_suffix(path.suffix+'.partial').write_bytes(r.content);path.with_suffix(path.suffix+'.partial').replace(path)
  else:
   chunk=64*1024*1024;parts=TARGET/'.download-parts';parts.mkdir(exist_ok=True)
   def fetch(i):
    lo=i*chunk;hi=min(size,lo+chunk)-1;part=parts/f'{i:04d}'
    if part.exists() and part.stat().st_size==hi-lo+1:return
    for attempt in range(4):
     try:
      with session().get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=(30,120)) as response:
       response.raise_for_status();assert response.status_code==206 and response.headers['Content-Range']==f'bytes {lo}-{hi}/{size}'
       with part.with_suffix('.partial').open('wb') as out:
        for block in response.iter_content(1024*1024):out.write(block)
      assert part.with_suffix('.partial').stat().st_size==hi-lo+1;part.with_suffix('.partial').replace(part);return
     except Exception:
      if attempt==3:raise
      time.sleep(2*(attempt+1))
   count=(size+chunk-1)//chunk
   with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for done,f in enumerate(concurrent.futures.as_completed([pool.submit(fetch,i) for i in range(count)]),1):
     f.result()
     if done%10==0 or done==count:print(json.dumps({'file':name,'parts':done,'total_parts':count,'elapsed_s':round(time.time()-start)}),flush=True)
   with path.with_suffix('.partial').open('wb') as out:
    for i in range(count):
     with (parts/f'{i:04d}').open('rb') as src:shutil.copyfileobj(src,out,16*1024*1024)
   path.with_suffix('.partial').replace(path)
 assert path.stat().st_size==size,(name,'size mismatch')
 h=hashlib.sha256() if 'lfs' in item else hashlib.sha1(f'blob {size}\0'.encode())
 with path.open('rb') as inp:
  for block in iter(lambda:inp.read(16*1024*1024),b''):h.update(block)
 expected=item['lfs']['sha256'] if 'lfs' in item else item['blobId'];assert h.hexdigest()==expected,(name,'hash mismatch')
 results.append({'path':name,'size':size,'hash':h.hexdigest(),'verified':True});print('VERIFIED',name,size,flush=True)
 if name=='model.safetensors':
  from safetensors import safe_open
  with safe_open(str(path),framework='np') as tensor_file:
   tensor_count=len(list(tensor_file.keys()));assert tensor_count>0
  if (TARGET/'.download-parts').exists():shutil.rmtree(TARGET/'.download-parts')
report={'complete':True,'repo_id':REPO,'revision':REV,'files':results,'tensor_count':tensor_count,'elapsed_s':time.time()-start,'evaluation_started':False}
(BASE/'manifests/simplemem-robomme-integrity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
