"""Wait for an idle GPU, infer once, then render on CPU. Never train or stop other jobs."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
OUT = Path('/cache/cvpzhr/result/第一次实验/评估结果/事件动态合并-20261103')
PYTHON = '/cache/cvpzhr/envs/forest/bin/python'
CODE = '/cache/cvpzhr/code/forest-train3-20261031-0009'


def status(phase, **values):
    payload = dict(phase=phase, pid=os.getpid(), updated_at=time.time(), output=str(OUT), **values)
    tmp = ROOT/'status.tmp';tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2));tmp.replace(ROOT/'status.json')


def idle_cards():
    args = ['nvidia-smi','--format=csv,noheader,nounits']
    raw = subprocess.check_output(args+['--query-gpu=index,uuid,utilization.gpu,memory.used'],text=True)
    apps = subprocess.check_output(args+['--query-compute-apps=gpu_uuid,pid'],text=True)
    busy = {line.split(',')[0].strip() for line in apps.splitlines()}
    cards = []
    for line in raw.splitlines():
        index, uuid, util, memory = [v.strip() for v in line.split(',')]
        if uuid not in busy and int(util)<=5 and int(memory)<1024:
            cards.append(int(index))
    return cards


def launch(script, gpu):
    env = {**os.environ,'PYTHONPATH':CODE,'CUDA_VISIBLE_DEVICES':str(gpu) if gpu is not None else '',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1',
        'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4'}
    with (ROOT/(script+'.log')).open('a') as log:
        result = subprocess.run([PYTHON,str(ROOT/script)],env=env,cwd=CODE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{script} exited {result.returncode}; see {script}.log')


def main():
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'status.json').exists() and json.loads((ROOT/'status.json').read_text())['phase']=='completed':
            return
        try:
            if not (OUT/'trace.json').exists():
                while True:
                    status('waiting_for_idle_gpu')
                    free=idle_cards()
                    if free:
                        time.sleep(15)
                        verified=set(free)&set(idle_cards())
                        if verified:
                            gpu=min(verified)
                            break
                    time.sleep(45)
                if shutil.disk_usage(ROOT).free<3*2**30:
                    raise RuntimeError('Less than 3 GiB free; no files removed')
                status('inference',physical_gpu=gpu)
                launch('infer.py',gpu)
            status('rendering')
            launch('render.py',None)
            status('completed',report=str(OUT/'report.json'))
        except BaseException as error:
            (ROOT/'failure.json').write_text(json.dumps(dict(error=str(error),traceback=traceback.format_exc()),ensure_ascii=False,indent=2))
            status('failed',error=str(error))
            raise


if __name__=='__main__':
    main()
