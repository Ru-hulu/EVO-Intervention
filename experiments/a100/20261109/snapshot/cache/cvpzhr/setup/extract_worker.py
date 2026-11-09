from pathlib import Path
import hashlib
import json
import sys
import tarfile
import time

root=Path('/cache/cvpzhr')
sys.path.insert(0,str(root/'setup/compression'))
import zstandard as z
transfers=root/'setup/transfers'
previous=root/'setup/extract-status.json'
status=json.loads(previous.read_text()) if previous.exists() else {'started':time.time(),'completed':[]}
status['phase']='extracting'
status.pop('error',None)
done={x['filename'].replace('.tar.zst','.json') for x in status['completed']}
def save():
    tmp=root/'setup/extract-status.json.tmp';tmp.write_text(json.dumps(status,ensure_ascii=False,indent=2));tmp.replace(root/'setup/extract-status.json')
try:
    save()
    while True:
        for meta in sorted(transfers.glob('[0-9][0-9][0-9].json')):
            if meta.name in done:continue
            record=json.loads(meta.read_text())
            path=transfers/record['filename']
            with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==record['sha256']
            with path.open('rb') as f,z.ZstdDecompressor(max_window_size=2**30).stream_reader(f) as reader,tarfile.open(fileobj=reader,mode='r|') as archive:
                archive.extractall(root/'datasets/forest-5fps',filter='data')
            done.add(meta.name)
            status['completed'].append(record);save()
            path.unlink()
            print('EXTRACTED',record['entry'],flush=True)
        complete=transfers/'complete.json'
        if complete.exists() and len(done)==json.loads(complete.read_text())['total']:
            status.update(phase='completed',finished=time.time());save();break
        time.sleep(5)
except Exception as exc:
    status.update(phase='failed',error=repr(exc));save();raise
