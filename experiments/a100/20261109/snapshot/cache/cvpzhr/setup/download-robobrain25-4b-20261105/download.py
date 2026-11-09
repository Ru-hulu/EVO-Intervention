"""Download official BAAI weights directly on this host; no local relay or GPU use."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import fcntl, hashlib, json, os, time, traceback
from urllib.parse import quote
import requests

ROOT=Path(__file__).resolve().parent
TARGET=Path('/cache/cvpzhr/models/RoboBrain2.5-4B')
REPO='BAAI/RoboBrain2.5-4B'
REVISION='ec4b3bf5e969b48ad17df432b479c10402a0079b'
HF='https://hf-mirror.com'
MS='https://modelscope.cn/api/v1/models/'+REPO
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write(path,d):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)
def status(phase,**kw):
    write(ROOT/'status.json',dict(phase=phase,pid=os.getpid(),updated_at=now(),repo=REPO,revision=REVISION,target=str(TARGET),**kw))
def metadata(url):
    for attempt in range(4):
        try:
            r=requests.get(url,timeout=(15,45));r.raise_for_status();return r.json()
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(3*(attempt+1))
def verify(path,item):
    if not path.exists() or path.stat().st_size!=item['size']:return None
    sha=hashlib.sha256()
    git=hashlib.sha1(('blob '+str(item['size'])+'\0').encode())
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b);git.update(b)
    ok=(sha.hexdigest()==item['lfs']['sha256']) if 'lfs' in item else git.hexdigest()==item['blobId']
    return sha.hexdigest() if ok else None
def main():
    lock=(ROOT/'download.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status('resolving_metadata')
    hf=metadata(HF+'/api/models/'+REPO+'/revision/'+REVISION+'?blobs=true')
    assert hf['sha']==REVISION and hf['id']==REPO
    write(ROOT/'huggingface-metadata.json',hf)
    try:
        ms=metadata(MS+'/repo/files?Revision=master&Recursive=true')
        write(ROOT/'modelscope-metadata.json',ms)
        msfiles={x['Path']:x for x in ms.get('Data',{}).get('Files',[]) if x.get('Type')=='blob'}
    except Exception as e:
        print('ModelScope metadata unavailable:',repr(e),flush=True);msfiles={}
    files=hf['siblings']
    TARGET.mkdir(parents=True,exist_ok=True)
    verified=[]
    for item in files:
        name=item['rfilename']
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        target=TARGET/name;target.parent.mkdir(parents=True,exist_ok=True)
        digest=verify(target,item)
        if digest:
            verified.append(dict(name=name,sha256=digest,size=item['size'],reused=True));continue
        if target.exists():raise RuntimeError('Unverified existing file: '+str(target))
        partial=target.with_name(target.name+'.part')
        urls=[]
        m=msfiles.get(name)
        if m and m['Size']==item['size'] and ('lfs' not in item or m['Sha256']==item['lfs']['sha256']):
            urls.append(MS+'/repo?Revision='+quote(m['Revision'],safe='')+'&FilePath='+quote(name,safe=''))
        urls.append(HF+'/'+REPO+'/resolve/'+REVISION+'/'+quote(name,safe='/')+'?download=true')
        errors=[];finished=False
        for url in urls:
            for attempt in range(4):
                try:
                    offset=partial.stat().st_size if partial.exists() else 0
                    if offset==item['size']:
                        status('verifying',file=name,completed_files=len(verified),total_files=len(files))
                        digest=verify(partial,item)
                        if digest:finished=True;break
                        partial.replace(partial.with_name(partial.name+'.invalid-'+str(time.time_ns())))
                        offset=0
                    status('downloading',file=name,bytes=offset,file_bytes=item['size'],completed_files=len(verified),total_files=len(files))
                    headers={'Accept-Encoding':'identity'}
                    if offset:headers['Range']='bytes='+str(offset)+'-'
                    with requests.get(url,headers=headers,stream=True,timeout=(20,90)) as response:
                        response.raise_for_status()
                        if response.status_code==206:
                            assert response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')
                        elif response.status_code==200:offset=0
                        else:raise RuntimeError('Unexpected HTTP '+str(response.status_code))
                        last=time.monotonic()
                        with partial.open('ab' if offset else 'wb') as f:
                            for chunk in response.iter_content(4*1024*1024):
                                if not chunk:continue
                                f.write(chunk);offset+=len(chunk)
                                if time.monotonic()-last>15:
                                    f.flush();status('downloading',file=name,bytes=offset,file_bytes=item['size'],completed_files=len(verified),total_files=len(files))
                                    last=time.monotonic()
                    status('verifying',file=name,completed_files=len(verified),total_files=len(files))
                    digest=verify(partial,item)
                    if not digest:raise RuntimeError('Size or checksum mismatch for '+name)
                    finished=True;break
                except Exception as e:
                    errors.append(repr(e));print(name,'attempt',attempt+1,repr(e),flush=True)
                    if attempt<3:time.sleep(5*(attempt+1))
            if finished:break
        if not finished:raise RuntimeError(name+': '+str(errors))
        partial.replace(target)
        verified.append(dict(name=name,size=item['size'],sha256=digest,source=url))
        write(ROOT/'verified-files.json',verified)
        print('Verified',name,item['size'],flush=True)
    from transformers import AutoConfig, AutoProcessor
    config=AutoConfig.from_pretrained(TARGET,local_files_only=True)
    processor=AutoProcessor.from_pretrained(TARGET,local_files_only=True)
    assert config.model_type=='qwen3_vl'
    receipt=dict(completed_at=now(),repo=REPO,revision=REVISION,target=str(TARGET),
                 files=verified,total_bytes=sum(x['size'] for x in verified),
                 architecture=config.architectures,processor=type(processor).__name__,
                 checksums_verified_against_huggingface=True,gpu_used=False)
    write(ROOT/'complete.json',receipt)
    status('completed',files=len(verified),total_bytes=receipt['total_bytes'],receipt=str(ROOT/'complete.json'))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('failed',error=repr(e),traceback=traceback.format_exc())
        raise
