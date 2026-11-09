import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')
CODE='/cache/cvpzhr/code/forest-train3-20261031-0009'
def idle():
    pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    return not pids and all(int(row.split(',')[0])<=5 and int(row.split(',')[1])<1024 for row in rows)

while not (RUN/'benchmarks/ddp-pad64-b64-gc1-w4.json').exists():time.sleep(15)
while not idle():time.sleep(10)
time.sleep(15);assert idle()
env=os.environ.copy();env.update(PYTHONPATH=CODE,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
name='probe-pad64-b6-gc0'
with (RUN/'benchmarks'/f'{name}.log').open('w') as log:
    rc=subprocess.run([sys.executable,str(RUN/'benchmark.py'),'--mode','probe','--batch','6','--name',name],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
result=json.loads((RUN/'benchmarks'/f'{name}.json').read_text());print(json.dumps(result),flush=True)
if rc==0 and result.get('status')=='passed' and max(result['peak_reserved_gib'])<70:
    time.sleep(15);assert idle()
    name='ddp-pad64-b6-gc0-w4';env['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    with (RUN/'benchmarks'/f'{name}.log').open('w') as log:
        rc=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
            str(RUN/'benchmark.py'),'--batch','6','--workers','4','--name',name,'--replay'],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if rc:raise RuntimeError('Batch 6 DDP check failed; inspect log')
    print((RUN/'benchmarks'/f'{name}.json').read_text(),flush=True)
(RUN/'extra-candidate-done.json').write_text(json.dumps(dict(status='completed'))+'\n')
