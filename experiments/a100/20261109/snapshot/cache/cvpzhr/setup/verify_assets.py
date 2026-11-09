import hashlib
import json
from pathlib import Path

root=Path('/cache/cvpzhr')
manifest=json.loads((root/'setup/source-manifest.json').read_text())
roots={'model':root/'models/Qwen3.5-4B','code':root/'code/forest'}
for name,expected in manifest.items():
    label,relative=name.split('/',1)
    path=roots[label]/relative
    assert path.stat().st_size==expected['size'],name
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    assert digest.hexdigest()==expected['sha256'],name
report={'status':'PASS','files':len(manifest),'all_hashes_verified':True}
(root/'setup/assets-check.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
