"""Run the isolated four-GPU weighted experiment and archive final results."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

WORK=Path('/cache/cvpzhr/setup/forest-s01-critical-sampling-20261106')
OUT=Path('/cache/cvpzhr/outputs/forest-s01-critical-sampling-20261106')
ARCHIVE=Path('/cache/cvpzhr/experiments/forest-s01-decline/02-critical-sampling-20261106')


def status(phase,**extra):
    value=dict(phase=phase,updated_at=time.time(),controller_pid=os.getpid(),output_dir=str(OUT),**extra)
    p=WORK/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def finish(returncode):
    manifest=json.loads((ARCHIVE/'manifest.json').read_text())
    manifest.update(status='completed' if returncode==0 else 'failed',returncode=returncode,finished_at=time.time())
    for f in list(OUT.glob('*.pt'))+list(OUT.glob('*.jsonl'))+list(OUT.glob('config.json')):
        dst=ARCHIVE/f.name
        if f.suffix=='.pt':
            if not dst.exists():os.link(f,dst)
        else:shutil.copyfile(f,dst)
    validation=OUT/'validation.jsonl'
    if validation.exists():
        vals=[json.loads(l) for l in validation.read_text().splitlines()]
        manifest['final_validation']=vals[-1];manifest['best_validation']=min(vals,key=lambda v:v['mae_pp'])
    (ARCHIVE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    if (OUT/'train.jsonl').exists() and validation.exists():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows=[json.loads(l) for l in (OUT/'train.jsonl').read_text().splitlines()]
        fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
        loss=[r['loss'] for r in rows];steps=[r['step'] for r in rows]
        axes[0].plot(steps,loss,alpha=.2,color='#137e79')
        axes[0].plot(steps,[sum(loss[max(0,i-19):i+1])/len(loss[max(0,i-19):i+1]) for i in range(len(loss))],label='Train CE, 20-record mean',color='#137e79')
        axes[0].plot([v['step'] for v in vals],[v['loss'] for v in vals],'o-',label='Full-val CE',color='#d97326')
        axes[0].set(ylabel='Cross-entropy',title='Critical-sampling fine-tuning');axes[0].legend()
        for key in ['mae_pp','rmse_pp']:axes[1].plot([v['step'] for v in vals],[v[key] for v in vals],'o-',label=key)
        axes[1].set(xlabel='Optimizer step',ylabel='Full-val error (pp)');axes[1].legend()
        fig.savefig(ARCHIVE/'loss.png',dpi=160)


def main():
    import fcntl
    lock=(WORK/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(used)==4 and all(int(x)<1000 for x in used), 'GPU conflict: leave other jobs untouched'
    assert not OUT.exists(), 'Do not append to another run'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONPATH=str(WORK/'code'),OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    with (WORK/'train.log').open('w') as log:
        p=subprocess.Popen(['/cache/cvpzhr/envs/forest/bin/python','-u','-m','torch.distributed.run',
            '--standalone','--nproc_per_node=4','-m','mem_lite.train_stage3','--config',str(WORK/'config.json')],
            cwd=WORK,env=env,stdout=log,stderr=subprocess.STDOUT)
    manifest=json.loads((ARCHIVE/'manifest.json').read_text());manifest.update(status='running',started_at=time.time(),training_pid=p.pid)
    (ARCHIVE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    while p.poll() is None:
        log=OUT/'train.jsonl';last=json.loads(log.read_text().splitlines()[-1]) if log.exists() and log.stat().st_size else {}
        status('training' if last else 'loading',training_pid=p.pid,last_step=last.get('step'),last_loss=last.get('loss'))
        time.sleep(10)
    status('archiving',returncode=p.returncode)
    finish(p.returncode)
    status('completed' if p.returncode==0 else 'training_failed',returncode=p.returncode)


if __name__=='__main__':main()
