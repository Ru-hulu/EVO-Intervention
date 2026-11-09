"""Publish the ten-trajectory result after inference finishes, preserving old entries."""
import json
import subprocess
import time
from pathlib import Path

root=Path('/cache/cvpzhr/evaluations/forest-s01-val10-step13224-20261106')
while not (root/'metrics.json').exists():
    log=(root/'eval.log').read_text()
    if 'Traceback (most recent call last)' in log: raise RuntimeError(log[-3000:])
    time.sleep(5)
report=json.loads((root/'metrics.json').read_text())
selection=json.loads((root/'selection.json').read_text())['episodes']
assert len(selection)==10
for model in ['fa','rec']:
    rows=[r for r in selection if r['category'].startswith(model+'_')]
    assert len(rows)==5 and sum(r['task_success'] for r in rows)==2
    subprocess.run(['/cache/cvpzhr/envs/forest/bin/python',
        '/cache/cvpzhr/setup/forest-s01-decline-20261106-4gpu/plot_five.py',str(root),'--model',model],check=True)
link=Path('/cache/cvpzhr/evaluations/forest-s01-val5-20261106/viewer-data')/report['dataset']
if not link.exists():link.symlink_to(root/'viewer-data'/report['dataset'],target_is_directory=True)
print(json.dumps(report,ensure_ascii=False),flush=True)
