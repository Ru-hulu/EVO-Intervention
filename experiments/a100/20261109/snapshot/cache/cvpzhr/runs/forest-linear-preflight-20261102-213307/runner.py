import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

RUN=Path(__file__).resolve().parent
CODE=Path('/cache/cvpzhr/code/forest-linear-preflight-ad785a4-20261102-213307')
PY='/cache/cvpzhr/envs/forest/bin/python'


def status(phase,**kwargs):
    data=dict(phase=phase,pid=os.getpid(),timestamp=time.time(),code=str(CODE),formal_training=False,**kwargs)
    p=RUN/'status.tmp';p.write_text(json.dumps(data,ensure_ascii=False,indent=2));p.replace(RUN/'status.json')
    print(json.dumps(data,ensure_ascii=False),flush=True)


def snapshot():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(gpus=[r.strip().split(', ') for r in rows.splitlines()],apps=[r.strip().split(', ') for r in apps.splitlines()])


def idle(gpus):
    clear=0
    for attempt in range(5):
        snap=snapshot()
        ready=True
        for row in snap['gpus']:
            if row[0] in gpus:
                assert not any(p[0]==row[1] for p in snap['apps']),f'GPU busy: {snap}'
                ready=ready and int(row[2])<1024 and int(row[3])<=5
        with (RUN/'gpu.jsonl').open('a') as f: f.write(json.dumps(dict(time=time.time(),**snap))+'\n')
        clear=clear+1 if ready else 0
        if clear==2: return
        time.sleep(15)
    raise RuntimeError('GPU has not returned to idle within the preflight wait')


def stage(name,args,gpus):
    if gpus: idle(gpus)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(gpus),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    status(name,selected_gpus=gpus)
    with (RUN/(name+'.log')).open('w') as log:
        process=subprocess.Popen(args,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(name,child_pid=process.pid,selected_gpus=gpus)
        while process.poll() is None:
            snap=snapshot()
            with (RUN/'gpu.jsonl').open('a') as f: f.write(json.dumps(dict(time=time.time(),phase=name,**snap))+'\n')
            # One compute process per selected GPU; do not touch any other job.
            for row in snap['gpus']:
                if row[0] in gpus and sum(p[0]==row[1] for p in snap['apps'])>1:
                    process.terminate()
                    process.wait(timeout=60)
                    raise RuntimeError(f'Another GPU process appeared on GPU {row[0]}')
            time.sleep(5)
    return process.returncode


def main():
    with (RUN/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            rc=stage('cpu-tests',[PY,'-m','pytest','mem_lite/tests','-q'],[])
            if rc: raise RuntimeError(f'CPU tests failed: {rc}')
            rc=stage('functional',[PY,str(RUN/'preflight.py'),'functional','--output',str(RUN/'functional')],['1'])
            if rc: raise RuntimeError(f'Functional test failed: {rc}')
            rc=stage('capacity-b4',[PY,str(RUN/'preflight.py'),'capacity','--output',str(RUN/'capacity-b4')],['1'])
            checkpointing=False
            if rc:
                failure=RUN/'capacity-b4/failure-rank-0.json'
                record=json.loads(failure.read_text()) if failure.exists() else {}
                if record.get('type')!='OutOfMemoryError': raise RuntimeError(f'Capacity test failed: {record}')
                checkpointing=True
                rc=stage('capacity-b4-checkpoint',[PY,str(RUN/'preflight.py'),'capacity','--checkpointing','--output',str(RUN/'capacity-b4-checkpoint')],['1'])
                if rc: raise RuntimeError(f'Capacity with checkpointing failed: {rc}')
            args=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',str(RUN/'preflight.py'),'capacity','--output',str(RUN/'ddp-b4')]
            if checkpointing: args.append('--checkpointing')
            rc=stage('ddp-b4',args,['1','2','3'])
            if rc: raise RuntimeError(f'DDP check failed: {rc}')
            report=dict(commit='ad785a4128ed0862742e888338349f4d42b27dff',formal_training=False,
                        recommended_batch_per_gpu=4,sequence_window=8,gradient_checkpointing=checkpointing,
                        functional=json.loads((RUN/'functional/functional.json').read_text()),
                        single=json.loads((RUN/('capacity-b4-checkpoint' if checkpointing else 'capacity-b4')/'summary.json').read_text()),
                        ddp=[json.loads((RUN/f'ddp-b4/rank-{i}.json').read_text()) for i in range(3)])
            (RUN/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            status('completed',report=str(RUN/'report.json'))
        except BaseException as e:
            failure=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
            (RUN/'failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2))
            status('failed',error=str(e))
            raise


if __name__=='__main__': main()
