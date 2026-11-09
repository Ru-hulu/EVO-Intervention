from pathlib import Path
import json
import subprocess
import sys
import time

root=Path('/cache/cvpzhr')
while True:
    status=json.loads((root/'setup/extract-status.json').read_text())
    if status['phase']=='failed':raise RuntimeError(status['error'])
    if status['phase']=='completed':break
    time.sleep(10)
subprocess.run([sys.executable,str(root/'setup/check_data.py')],check=True)
subprocess.run([sys.executable,str(root/'setup/finalize.py')],check=True)
