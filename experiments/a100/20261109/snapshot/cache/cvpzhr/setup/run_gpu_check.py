import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/cache/cvpzhr')
with (root/'setup/gpu-check.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for i in range(2):
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if processes: raise RuntimeError('GPU has existing compute processes; check not started: '+processes)
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
        assert len(rows)==4
        for row in rows:
            index,memory,util=map(int,row.split(','))
            if memory>=1024 or util>5: raise RuntimeError('GPU busy: '+row)
        if i==0: time.sleep(15)
    script=sys.argv[1]
    if script not in ['nccl_check.py','preflight.py']: raise ValueError(script)
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(root/'setup'/script)],cwd=root/'code/forest',check=True)
