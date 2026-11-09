import hashlib
import json
from pathlib import Path
import zipfile
from huggingface_hub import snapshot_download

base=Path('/cache/robot-baselines')
manifest=json.loads((base/'manifests/robomme-perceptual-framesamp-modul-source.json').read_text())
target=base/'models/RoboMME/perceptual-framesamp-modul'
print('Downloading', manifest['repo_id'], manifest['revision'], flush=True)
snapshot_download(repo_id=manifest['repo_id'], revision=manifest['revision'], endpoint=manifest['endpoint'], local_dir=target, max_workers=4)
verified=[]
for item in manifest['files']:
    path=target/item['path']
    assert path.stat().st_size==item['size'], str(path)
    if item['sha256']:
        h=hashlib.sha256()
    else:
        h=hashlib.sha1()
        h.update(f"blob {item['size']}\0".encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024), b''):
            h.update(block)
    assert h.hexdigest()==(item['sha256'] or item['blob_id']), str(path)
    verified.append(item['path'])
    print('HASH_OK', item['path'], flush=True)
with zipfile.ZipFile(target/'79999.zip') as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    bad=z.testzip()
    assert bad is None, bad
    print('ZIP_CRC_OK', len(z.infolist()), 'entries; extracting', flush=True)
    z.extractall(target)
print('EXTRACTED', flush=True)
files=[str(p.relative_to(target)) for p in target.rglob('*') if p.is_file() and '.cache' not in p.parts]
report=dict(repo_id=manifest['repo_id'],revision=manifest['revision'],directory=str(target),verified=verified,zip_crc_ok=True,files=files)
(base/'manifests/robomme-perceptual-framesamp-modul-integrity.json').write_text(json.dumps(report,indent=2)+'\n')
print('DOWNLOAD_VERIFIED_AND_EXTRACTED', str(target), flush=True)
