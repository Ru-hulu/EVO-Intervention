"""Copy the verified runtime into an independent environment on the system disk."""
from pathlib import Path
import shutil
import sys

assert sys.prefix == '/root/envs/robometer'
source = Path('/root/autodl-tmp/envs/procvlm/lib/python3.12/site-packages')
target = Path(sys.prefix) / 'lib/python3.12/site-packages'
stamp = Path(sys.prefix) / 'runtime-copy-complete'
if not stamp.exists():
    shutil.copytree(source, target, dirs_exist_ok=True, symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__'))
    stamp.write_text('Independent package files copied from the verified CUDA runtime.\n')
print('Independent CUDA runtime ready.', flush=True)
