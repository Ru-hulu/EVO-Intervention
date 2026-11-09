import json
import os
from pathlib import Path
import subprocess
import sys
import time

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')
CODE='/cache/cvpzhr/code/forest-train3-20261031-0009'

def idle():
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    return not processes and len(rows)==4 and all(int(x.split(',')[0])<=5 and int(x.split(',')[1])<1024 for x in rows)

def main():
    while not (RUN/'padding-check.json').exists():time.sleep(10)
    assert json.loads((RUN/'padding-check.json').read_text())['status']=='passed'
    probes=['probe-b4-gc0','probe-b64-gc1','probe-b96-gc1','probe-b8-gc1']
    while not all((RUN/'benchmarks'/f'{name}.json').exists() for name in probes):time.sleep(10)
    while not idle():time.sleep(10)
    time.sleep(15)
    assert idle(),'Another GPU process appeared'
    for batch,gc in [(4,False),(8,True),(16,True),(32,True),(64,True)]:
        probe=RUN/'benchmarks'/f'probe-b{batch}-gc{int(gc)}.json'
        if probe.exists() and json.loads(probe.read_text()).get('status')!='passed':continue
        name=f'ddp-pad64-b{batch}-gc{int(gc)}-w4'
        env=os.environ.copy();env.update(PYTHONPATH=CODE,OMP_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='0,1,2,3')
        args=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(RUN/'benchmark.py'),
              '--batch',str(batch),'--workers','4','--name',name,'--replay']
        if gc:args.append('--checkpointing')
        print(json.dumps(dict(name=name,phase='starting',time=time.time())),flush=True)
        with (RUN/'benchmarks'/f'{name}.log').open('w') as log:
            result=subprocess.run(args,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(name=name,returncode=result.returncode,time=time.time())),flush=True)
        if result.returncode:raise RuntimeError(f'{name} failed: inspect its log')
        print((RUN/'benchmarks'/f'{name}.json').read_text(),flush=True)
        time.sleep(5)
        if not idle():
            time.sleep(15)
            assert idle(),'Other GPU process appeared'

if __name__=='__main__':main()
