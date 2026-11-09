from pathlib import Path
import json
import subprocess
import sys
import time

root=Path('/cache/cvpzhr')
manifest=json.loads((root/'setup/source-manifest.json').read_text())
models={name.split('/',1)[1]:item for name,item in manifest.items() if name.startswith('model/')}
while not all((root/'models/Qwen3.5-4B'/name).exists() and (root/'models/Qwen3.5-4B'/name).stat().st_size==item['size'] for name,item in models.items()):
    time.sleep(10)
subprocess.run([sys.executable,str(root/'setup/verify_assets.py')],check=True)
subprocess.run([sys.executable,str(root/'setup/run_gpu_check.py'),'preflight.py'],check=True)
