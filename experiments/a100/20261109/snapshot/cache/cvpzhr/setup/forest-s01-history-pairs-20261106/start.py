"""Prepare new caches, validate them, then launch the isolated four-GPU run."""
import os
from pathlib import Path
import subprocess

WORK=Path('/cache/cvpzhr/setup/forest-s01-history-pairs-20261106')
PYTHON='/cache/cvpzhr/envs/forest/bin/python'


if __name__=='__main__':
    env=dict(os.environ,PYTHONPATH=str(WORK/'code'),PYTHONUTF8='1',OMP_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    subprocess.run([PYTHON,str(WORK/'prepare.py'),'setup'],env=env,check=True)
    jobs=[]
    for gpu in range(4):
        with (WORK/f'cache-{gpu}.log').open('w') as log:
            jobs.append(subprocess.Popen([PYTHON,'-u',str(WORK/'cache.py'),'worker','--worker',str(gpu),'--workers','4'],
                env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in jobs];assert codes==[0]*4,codes
    subprocess.run([PYTHON,str(WORK/'cache.py'),'validate'],env=env,check=True)
    subprocess.run([PYTHON,str(WORK/'prepare.py'),'sampling'],env=env,check=True)
    subprocess.run([PYTHON,str(WORK/'launch.py')],env=env,check=True)
