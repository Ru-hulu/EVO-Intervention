import subprocess,json,time
from pathlib import Path
p=Path.cwd()
status=[]
for b in (8,10,11):
    with (p/f'benchmark-batch{b}.log').open('w') as log:
        code=subprocess.call(['/cache/cvpzhr/envs/robometer/bin/python','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(p/'benchmark.py'),'--config',str(p/'training.config.json'),'--batch',str(b)],stdout=log,stderr=subprocess.STDOUT)
    status.append({'batch':b,'returncode':code,'finished_at':time.time()})
    (p/'benchmark-status.json').write_text(json.dumps(status,indent=2))
    if code:break
