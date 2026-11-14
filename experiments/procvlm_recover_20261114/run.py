"""One-shot ProcVLM on GPU2/3 after verified FoREST and explicit queue handoff."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/cache/cvpzhr/setup/procvlm-recover-10-progress-only-e1-20261114')
OUTPUT=Path('/cache/cvpzhr/outputs/procvlm-recover-10-progress-only-2gpu-e1-20261114')
OTHER=Path('/cache/cvpzhr/setup/recover-five-finetune-20261114')
FOREST=Path('/cache/cvpzhr/setup/forest-recover-10-legacy-e1-20261114')


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def cards_free():
    data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    ids=[]
    for line in data.splitlines():
        i,u,m=[v.strip() for v in line.split(',')]
        if int(i) in (2,3):
            if int(m)>=1000:return False
            ids.append(u)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return len(ids)==2 and not any(u in processes for u in ids)


def handoff_ready():
    # The admission receipt is written only after checking the live controller,
    # not merely changing a plan that an old controller may already have loaded.
    p=ROOT/'handoff-verified.json'
    return p.is_file() and read(p).get('old_queue_cannot_launch_procvlm') is True


def run_child(config,pilot):
    assert cards_free(),'GPU2/3 belong to another active task'
    config_path=Path(config['setup_dir'])/'config.json';save(config_path,config)
    folder=Path(config['setup_dir'])/('pilot' if pilot else 'formal');folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'launch.json').exists(),'Refuse duplicate training launch'
    cmd=[config['python'],'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
         '-m','experiments.recover_finetune_20261114.train','--config',str(config_path)]
    if pilot:cmd+=['--pilot']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',PYTHONUNBUFFERED='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True',NCCL_DEBUG='WARN',TRITON_CACHE_DIR=str(ROOT/'triton-cache'))
    env.pop('PRM_CUDA_MEMORY_FRACTION',None);env.pop('FOREST_NATIVE_GATED_NORM',None)
    with (folder/'train.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT,
                                 start_new_session=True,stdin=subprocess.DEVNULL)
    detail=dict(status='pilot' if pilot else 'training',pid=process.pid,runner_pid=os.getpid(),
        gpus=[2,3],model='procvlm-2b',config=str(config_path),started=time.time(),
        microbatch=config['microbatch'],accumulation=config['accumulation'],effective_batch=16,
        epochs=1,folder=str(folder),command=cmd)
    save(folder/'launch.json',detail);save(ROOT/'status.json',detail)
    code=process.wait();detail.update(returncode=code,finished=time.time());save(folder/'exit.json',detail)
    return code,folder,detail


def wait_free():
    until=time.time()+120
    while not cards_free():
        assert time.time()<until,'CUDA context was not released after this task ended'
        time.sleep(2)


def main():
    lock=(ROOT/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/'runner.pid').write_text(str(os.getpid())+'\n')
    assert not OUTPUT.exists(),'Preserve existing formal training output'
    save(ROOT/'status.json',dict(status='waiting_for_forest_and_handoff',pid=os.getpid(),gpus=[2,3],updated=time.time()))
    while True:
        forest=read(FOREST/'training-status.json')
        assert forest['status']!='failed','FoREST failed; do not silently proceed'
        if forest['status']=='completed' and forest['final_checkpoint_verified'] and handoff_ready() and cards_free():break
        time.sleep(2)
    config=read(ROOT/'reference.config.json')
    assert config['name']=='procvlm-2b' and config['epochs']==1
    assert config['samples']==2550 and config['episodes']==10
    assert sha(config['train_data'])==config['train_sha256']
    assert read(ROOT/'reference-input-verification.json')['status']=='passed'
    config.update(gpus=[2,3],output_dir=str(OUTPUT))
    save(ROOT/'forest-dependency.json',forest)
    # Keep global batch16 and the exact native inputs. Only reduce microbatch or
    # enable activation checkpointing if the longest mixed batch needs more memory.
    chosen=None;trials=[]
    for micro,accumulation,checkpointing in [(2,4,False),(1,8,False),(1,8,True)]:
        trial=copy.deepcopy(config);name=f'b{micro}-gc{int(checkpointing)}'
        trial.update(microbatch=micro,accumulation=accumulation,effective_batch=16,
            gradient_checkpointing=checkpointing,setup_dir=str(ROOT/'trials'/name))
        code,folder,detail=run_child(trial,True)
        if code==0:
            result=read(folder/'verification.json');assert result['status']=='passed'
            assert result['peak_reserved_gib']<63 and result['replicas_equal']
            chosen=trial;trials.append(dict(name=name,status='passed',verification=result));break
        log=(folder/'train.log').read_text().lower()
        if 'out of memory' not in log and 'gpu headroom' not in log:
            raise RuntimeError(f'{name}: non-memory failure; see {folder}/train.log')
        trials.append(dict(name=name,status='memory_limit',returncode=code));save(ROOT/'trials.json',trials)
        wait_free()
    assert chosen is not None,'All bounded ProcVLM memory trials failed'
    save(ROOT/'trials.json',trials);wait_free()
    chosen['setup_dir']=str(ROOT/'procvlm-2b');save(ROOT/'chosen-config.json',chosen)
    code,folder,detail=run_child(chosen,False)
    if code:
        detail['status']='failed';save(ROOT/'status.json',detail);return code
    verification=read(folder/'verification.json')
    assert verification['status']=='passed' and verification['epoch']==1
    assert verification['expected_episodes']==10 and verification['expected_samples']==2550
    assert sha(chosen['train_data'])==chosen['train_sha256']
    detail.update(status='completed',verification=verification,checkpoint=verification['checkpoint'],
        source_pt_and_pixels_unchanged=True,progress_only=True)
    save(ROOT/'status.json',detail);return 0


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        save(ROOT/'status.json',dict(status='failed',error=traceback.format_exc(),updated=time.time()))
        raise
