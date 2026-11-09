"""Launch and monitor the migrated four-GPU continuation."""
import json
import os
from pathlib import Path
import subprocess
import time

work=Path('/cache/cvpzhr/setup/forest-s01-decline-20261106-4gpu')
out=Path('/cache/cvpzhr/outputs/forest-s01-decline-finetune-20261106-4gpu')

def status(phase,**extra):
    value=dict(phase=phase,updated_at=time.time(),controller_pid=os.getpid(),output_dir=str(out),**extra)
    p=work/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main():
    import fcntl
    lock=(work/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH=str(work/'code'),
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    with (work/'train.log').open('a') as log:
        p=subprocess.Popen(['/cache/cvpzhr/envs/forest/bin/python','-u','-m','torch.distributed.run',
            '--standalone','--nproc_per_node=4','-m','mem_lite.train_stage3','--resume',str(work/'resume-4gpu.pt')],
            cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
    while p.poll() is None:
        rows=(out/'train.jsonl').read_text().splitlines()
        last=json.loads(rows[-1]);status('training',training_pid=p.pid,last_step=last['step'],last_loss=last['loss'])
        time.sleep(10)
    status('completed' if p.returncode==0 else 'training_failed',returncode=p.returncode)

if __name__=='__main__':main()
