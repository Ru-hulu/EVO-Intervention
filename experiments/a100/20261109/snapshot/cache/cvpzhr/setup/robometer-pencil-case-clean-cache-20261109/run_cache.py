import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

SETUP=Path('/cache/cvpzhr/setup/robometer-pencil-case-clean-cache-20261109')
OUTPUT=Path('/cache/cvpzhr/datasets/robometer-pencil-case-5fps-h7-current3-clean-20261109')
PY='/cache/cvpzhr/envs/robometer/bin/python'
THIRD='/cache/cvpzhr/code/robometer-current3-w8-e1-20261104/third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'
ENV={**os.environ,'PYTHONPATH':str(SETUP/'code')+':'+THIRD,'CUDA_VISIBLE_DEVICES':'',
     'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
     'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONNOUSERSITE':'1'}

def main():
    with (SETUP/'cache.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'pid':os.getpid(),'started_at':time.time(),'output_dir':str(OUTPUT),'augmentation':'none','training_started':False}
        def update(**values):
            state.update(values,updated_at=time.time())
            p=SETUP/'cache-controller.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(SETUP/'cache-controller.json')
        try:
            for phase,script,args in [('preflight','verify_labels.py',[]),('building','prepare_cache.py',['--workers','32']),('auditing','verify_labels.py',['--built'])]:
                update(phase=phase)
                with (SETUP/(phase+'.log')).open('w') as log:
                    subprocess.run([PY,'-u',str(SETUP/script),*args],env=ENV,cwd=SETUP,stdout=log,stderr=subprocess.STDOUT,check=True)
            audit=json.loads((OUTPUT/'label-audit.json').read_text());assert audit['status']=='passed'
            ready={'status':'ready','task':'zipper-pencil-case-inspection','augmentation':'none',
                   'train_episodes':28,'train_samples':10789,'validation_episodes':7,'validation_samples':2177,
                   'input_contract':'history7-main-current3-v1','training_started':False}
            (OUTPUT/'cache-ready.json').write_text(json.dumps(ready,indent=2))
            update(phase='completed',returncode=0,elapsed_seconds=time.time()-state['started_at'])
        except BaseException as exc:
            update(phase='failed',error=repr(exc));raise

if __name__=='__main__':main()
