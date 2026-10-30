"""Copy the working CUDA runtime into an independent venv without shared package files."""
from pathlib import Path
import shutil
import sys

target = Path(sys.prefix) / 'lib/python3.12/site-packages'
source = Path('/root/autodl-tmp/envs/mem-lite/lib/python3.12/site-packages')
assert sys.prefix == '/root/autodl-tmp/envs/procvlm'
stamp = Path(sys.prefix) / 'runtime-copy-complete'
if not stamp.exists():
    shutil.copytree(source, target, dirs_exist_ok=True, symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__'))
    stamp.write_text('Independent file copies of the verified CUDA runtime; no shared site-packages.\n')
print('Independent CUDA package copy ready.', flush=True)
