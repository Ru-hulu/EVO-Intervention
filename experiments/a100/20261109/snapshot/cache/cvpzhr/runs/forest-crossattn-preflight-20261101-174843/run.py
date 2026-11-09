import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

run=Path(__file__).resolve().parent
settings=json.loads((run/'settings.json').read_text())
code=Path(settings['code']); output=Path(settings['output']); python=settings['python']
env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','PYTHONPATH':str(code),'OMP_NUM_THREADS':'4',
     'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','WANDB_DISABLED':'true','PYTHONDONTWRITEBYTECODE':'1'}
def status(phase,**kw):
    (run/'status.json').write_text(json.dumps({'phase':phase,'updated_at':time.time(),**kw},indent=2))
def sample():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
    gpu={int(row[0]):row for row in [line.strip().split(', ') for line in raw.splitlines()]}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    used={line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    (run/'gpu-checks.jsonl').open('a').write(json.dumps({'time':time.time(),'gpus':raw,'processes':apps})+'\n')
    return all(gpu[i][1] not in used and int(gpu[i][2])<=5 and int(gpu[i][3])<1024 for i in (0,1))
def idle():
    for _ in range(12):
        if sample():
            time.sleep(15)
            if sample():return
        time.sleep(5)
    raise RuntimeError('Selected GPUs 0/1 are occupied; did not stop other jobs')
def execute(command,name):
    with (run/f'{name}.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            result=proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait(timeout=30)
            raise
    if result:raise RuntimeError(f'{name} exited {result}; see {name}.log')

with (run/'run.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        status('checking_idle',selected_gpus=[0,1]); idle()
        status('training_preflight',max_steps=12,selected_gpus=[0,1])
        execute([python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 '-m','mem_lite.train_stage3','--config',str(run/'train.json')],'training')
        status('checking_idle_before_validation'); idle()
        status('validation_preflight')
        execute([python,str(run/'validate.py'),str(output)],'validation')
        records=[json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
        assert records[-1]['step']==12
        report={'status':'passed','training_updates':12,'world_size':2,'batch_per_gpu':4,'formal_training_started':False,
                'four_gpu_test_pending':True,'training_records':records,
                'validation':json.loads((output/'preflight-validation.json').read_text())}
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        status('completed',output=str(output))
    except Exception:
        error=traceback.format_exc()
        (run/'failure.txt').write_text(error)
        status('failed',error=error)
        raise
