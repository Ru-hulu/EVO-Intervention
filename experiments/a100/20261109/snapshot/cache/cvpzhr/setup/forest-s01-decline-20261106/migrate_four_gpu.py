"""One-time 2-to-4 GPU migration preserving optimizer and global batch membership."""
import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import torch
from torch.utils.data import DistributedSampler, BatchSampler

SETUP=Path('/cache/cvpzhr/setup/forest-s01-decline-20261106')
OLD=Path('/cache/cvpzhr/outputs/forest-s01-decline-finetune-20261106')
NEW=OLD.with_name(OLD.name+'-4gpu')
WORK=SETUP.with_name(SETUP.name+'-4gpu')


def main():
    assert not NEW.exists() and not WORK.exists(), 'Refuse duplicate migration'
    ck=torch.load(OLD/'last.pt',map_location='cpu',weights_only=True)
    assert len(ck['rng_by_rank'])==2 and ck['config']['batch_size']==4
    assert ck['config']['gradient_accumulation_steps']==1 and ck['epoch']==2
    n=sum(sum(1 for l in open(p) if l.strip()) for p in ck['config']['sample_files'])
    assert n%2==0
    def indices(world,batch):
        ranks=[]
        for rank in range(world):
            sampler=DistributedSampler(range(n),num_replicas=world,rank=rank,seed=ck['config']['seed'],shuffle=True)
            sampler.set_epoch(ck['epoch'])
            ranks.append(list(BatchSampler(sampler,batch,False)))
        return ranks
    old,new=indices(2,4),indices(4,2)
    assert len(old[0])==len(new[0])==4408
    counted=0
    for b in range(len(old[0])):
        a=sorted(x for rank in old for x in rank[b])
        z=sorted(x for r,rank in enumerate(new) for i,x in enumerate(rank[b]) if r+(b*2+i)*4<n)
        assert a==z,(b,a,z)
        counted+=len(z)
    assert counted==n
    assert ck['step']==ck['epoch']*4408+ck['next_batch']
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert all(int(used[i])<1000 for i in (0,1)),used
    assert os.getpgid(1168652)==1166955
    WORK.mkdir();NEW.mkdir()
    # Retain the exact CPU-loaded state, independently of future last.pt replacements.
    torch.save(ck,WORK/'source-2gpu.pt')
    shutil.copytree(SETUP/'code',WORK/'code',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(SETUP/'train_stage3_four_gpu.py',WORK/'code/mem_lite/train_stage3.py')
    # Stop only the verified process group of this run, after preserving its checkpoint.
    os.killpg(1166955,signal.SIGTERM)
    for _ in range(30):
        time.sleep(1)
        output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if all(int(x)<1000 for x in output):break
    else:raise RuntimeError('Training process group did not release GPUs')
    for name in ['train.jsonl','validation.jsonl']:
        records=[json.loads(l) for l in (OLD/name).read_text().splitlines()]
        (NEW/name).write_text(''.join(json.dumps(r)+'\n' for r in records if r['step']<=ck['step']))
    for p in OLD.glob('*.pt'):
        if p.name!='last.pt':shutil.copyfile(p,NEW/p.name)
    cfg=ck['config'];cfg.update(batch_size=2,num_workers=2,validation_num_workers=2,output_dir=str(NEW))
    # Validation batch remains 4/GPU: doubles validation throughput without changing samples.
    rng=ck['rng_by_rank']
    for rank in (2,3):
        seed=cfg['seed']+ck['step']*4+rank
        rng.append(dict(torch=torch.Generator().manual_seed(seed).get_state(),
                        cuda=torch.Generator(device='cuda:0').manual_seed(seed).get_state()))
    ck['rng_by_rank']=rng
    audit=dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_step=ck['step'],
        epoch=ck['epoch'],next_batch=ck['next_batch'],old_world_size=2,new_world_size=4,
        global_batch=8,sampler_all_batches_equal=True,padding_excluded=True,train_samples=n,
        optimizer_preserved=True,randomness='old ranks 0/1 restored; new ranks 2/3 independently seeded',
        output_dir=str(NEW),code_dir=str(WORK/'code'))
    cfg['four_gpu_migration']=audit
    torch.save(ck,WORK/'resume-4gpu.pt')
    (WORK/'migration.json').write_text(json.dumps(audit,indent=2)+'\n')
    (SETUP/'status.json').write_text(json.dumps(dict(phase='migrated_to_four_gpus',**audit),indent=2)+'\n')
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
