import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')
CODE='/cache/cvpzhr/code/forest-train3-20261031-0009'
(RUN/'benchmarks').mkdir(exist_ok=True)

def idle():
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    return not processes and len(rows)==4 and all(int(x.split(',')[0])<=5 and int(x.split(',')[1])<1024 for x in rows)

assert idle(),'GPU not idle'
time.sleep(15)
assert idle(),'GPU no longer idle'
jobs=[]
for gpu,(batch,gc) in enumerate([(4,False),(64,True),(96,True),(8,True)]):
    name=f'probe-b{batch}-gc{int(gc)}'
    env=os.environ.copy();env.update(PYTHONPATH=CODE,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
    args=[sys.executable,str(RUN/'benchmark.py'),'--mode','probe','--batch',str(batch),'--name',name]
    if gc:args.append('--checkpointing')
    with (RUN/'benchmarks'/f'{name}.log').open('w') as log:
        p=subprocess.Popen(args,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
    jobs.append((name,p))
    print(json.dumps(dict(name=name,pid=p.pid,gpu=gpu)),flush=True)
for name,p in jobs:
    code=p.wait()
    print(json.dumps(dict(name=name,returncode=code)),flush=True)
    path=RUN/'benchmarks'/f'{name}.json'
    if path.exists():print(path.read_text(),flush=True)
