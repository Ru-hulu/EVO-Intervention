from pathlib import Path
import json, shutil
root=Path(__file__).resolve().parent
meta=json.loads((root/'media-manifest.json').read_text(encoding='utf-8'))
limits=json.loads((root/'axis_limits.json').read_text())
(root/'videos').mkdir(exist_ok=True)
for key,item in meta.items():
    if 'comparison' in key: continue
    source=root/'media'/key
    if not source.is_file(): raise FileNotFoundError(source)
    target=root/'videos'/item['filename']
    if not target.exists(): shutil.copy2(source,target)
    target.with_suffix('.json').write_text(json.dumps({'axis_limits':limits}),encoding='utf-8')
print('Prepared videos/ for stack_acc_comparisons.py')
