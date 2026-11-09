from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time
import requests

base=Path('/cache/robot-baselines')
manifest=json.loads((base/'manifests/robomme-perceptual-framesamp-modul-source.json').read_text())
target=base/'models/RoboMME/perceptual-framesamp-modul'
item=next(f for f in manifest['files'] if f['path']=='79999.zip')
url=f"{manifest['endpoint']}/{manifest['repo_id']}/resolve/{manifest['revision']}/79999.zip"
size=item['size']; chunk=64*1024*1024
parts=target/'.cache/range-parts';parts.mkdir(parents=True,exist_ok=True)

def fetch(i):
    start=i*chunk; end=min(size,start+chunk)-1
    out=parts/f'{i:04d}'
    if out.exists() and out.stat().st_size==end-start+1:return out
    for attempt in range(4):
        try:
            s=requests.Session();s.trust_env=False
            with s.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,120)) as r:
                r.raise_for_status()
                assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}', dict(r.headers)
                with out.with_suffix('.partial').open('wb') as f:
                    for block in r.iter_content(1024*1024):f.write(block)
            assert out.with_suffix('.partial').stat().st_size==end-start+1
            out.with_suffix('.partial').replace(out)
            return out
        except Exception:
            if attempt==3:raise
            time.sleep(2*(attempt+1))

count=(size+chunk-1)//chunk
start=time.time();done=0
with ThreadPoolExecutor(max_workers=16) as pool:
    for f in as_completed([pool.submit(fetch,i) for i in range(count)]):
        f.result();done+=1
        if done%8==0 or done==count:print(f'PARTS {done}/{count}; elapsed {time.time()-start:.0f}s',flush=True)
with (target/'79999.zip.partial').open('wb') as dst:
    for i in range(count):
        with (parts/f'{i:04d}').open('rb') as src:shutil.copyfileobj(src,dst,16*1024*1024)
(target/'79999.zip.partial').replace(target/'79999.zip')
print('RANGE_DOWNLOAD_DONE',flush=True)
# The official file hashes and archive CRC are checked by the companion script.
script=base/'setup/download-robomme-framesamp.py'
code=script.read_text().replace("snapshot_download(repo_id=manifest['repo_id'], revision=manifest['revision'], endpoint=manifest['endpoint'], local_dir=target, max_workers=4)", "assert all((target / f['path']).is_file() for f in manifest['files'])")
exec(compile(code,str(script),'exec'))
shutil.rmtree(parts)
for p in (target/'.cache/huggingface/download').glob('*.incomplete'):
    p.unlink()
