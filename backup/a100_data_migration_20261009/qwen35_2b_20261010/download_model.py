"""Detached ModelScope download with per-file SHA256 verification and resumable partials."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from urllib.parse import quote
import hashlib,json,time,traceback

AUDIT=Path('/data/evo-inter/verification/qwen35_2b_20261010')
DEST=Path('/data/evo-inter/models/Qwen3.5-2B')
FILES=[f for f in json.loads((AUDIT/'modelscope_manifest.json').read_text())['Data']['Files'] if f['Type']=='blob']

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def download(item):
 rel=PurePosixPath(item['Path'])
 if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe repository path')
 target=DEST.joinpath(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
 expected_size=item['Size'];expected_hash=item['Sha256']
 if target.exists():
  if target.stat().st_size==expected_size and digest(target)==expected_hash:return item['Path']
  raise RuntimeError('Refusing to replace an existing mismatched file: '+str(target))
 partial=target.with_name(target.name+'.download-part')
 url='https://modelscope.cn/models/Qwen/Qwen3.5-2B/resolve/master/'+quote(item['Path'],safe='/')
 for attempt in range(1,5):
  try:
   offset=partial.stat().st_size if partial.exists() else 0
   if offset>expected_size:raise RuntimeError('Partial file larger than expected')
   if offset<expected_size:
    headers={'User-Agent':'Evo-Inter-model-download/1.0'}
    if offset:headers['Range']='bytes='+str(offset)+'-'
    with urlopen(Request(url,headers=headers),timeout=90) as response:
     append=offset>0 and response.status==206
     if append and not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):
      raise RuntimeError('Incorrect HTTP range response')
     with partial.open('ab' if append else 'wb') as stream:
      for block in iter(lambda:response.read(4*1024*1024),b''):stream.write(block)
   if partial.stat().st_size!=expected_size:raise RuntimeError('File size mismatch')
   if digest(partial)!=expected_hash:
    partial.unlink()
    raise RuntimeError('SHA256 mismatch; discarding this downloader partial')
   partial.replace(target)
   print(json.dumps({'event':'file_verified','file':item['Path'],'bytes':expected_size}),flush=True)
   return item['Path']
  except Exception as exc:
   print(json.dumps({'event':'retry','file':item['Path'],'attempt':attempt,'error':str(exc)}),flush=True)
   if attempt==4:raise
   time.sleep(15*attempt)

def main():
 DEST.mkdir(parents=True,exist_ok=True)
 print(json.dumps({'event':'started','repo':'Qwen/Qwen3.5-2B','source':'ModelScope Qwen publisher','destination':str(DEST),'files':len(FILES),'bytes':sum(f['Size'] for f in FILES)}),flush=True)
 try:
  with ThreadPoolExecutor(max_workers=3) as pool:done=list(pool.map(download,FILES))
  result={'status':'complete','verified_files':done,'completed_unix':time.time()}
 except Exception as exc:
  result={'status':'failed','error':str(exc),'ended_unix':time.time()}
  traceback.print_exc()
  (AUDIT/'download_result.json').write_text(json.dumps(result,indent=2)+'\n')
  raise
 (AUDIT/'download_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)

if __name__=='__main__':main()
