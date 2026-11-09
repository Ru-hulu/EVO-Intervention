"""只编排冻结视觉推理和数据构建，不包含训练入口。"""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

CODE=Path('/cache/cvpzhr/code/forest-proprimem-20261030')
RUN=Path('/cache/cvpzhr/setup/proprimem-5fps-20261030')
OUT=Path('/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030')


def state(phase,**details):
    p=RUN/'status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),time=time.time(),**details),indent=2)+'\n');tmp.replace(p)


def idle():
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    return not processes and len(rows)==4 and all(int(x.split(',')[0])<=5 and int(x.split(',')[1])<1024 for x in rows)


def launch(command,name,gpu):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',PYTHONPATH=str(CODE))
    log=(RUN/f'{name}.log').open('a')
    process=subprocess.Popen([sys.executable,*command],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
    log.close()
    return process


RUN.mkdir(parents=True,exist_ok=True)
lock=(RUN/'run.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
children=[]
try:
    state('checking_gpu')
    if not idle():raise RuntimeError('GPU 非空闲；未启动构建')
    time.sleep(15)
    if not idle():raise RuntimeError('GPU 第二次检查非空闲；未启动构建')
    if not (OUT/'gpu-smoke.json').exists():
        state('gpu_smoke')
        p=launch([str(RUN/'smoke.py'),str(OUT)],'gpu-smoke',0);children=[p]
        if p.wait()!=0:raise RuntimeError('真实 GPU 预检失败，检查 gpu-smoke.log')
    children=[]
    # 子进程退出后，nvidia-smi 的利用率采样可能仍包含刚结束的预检。
    for attempt in range(30):
        if idle():break
        time.sleep(2)
    else:raise RuntimeError('GPU 尚未恢复空闲；暂停全量构建')
    time.sleep(15)
    if not idle():raise RuntimeError('GPU 第二次检查非空闲；暂停全量构建')
    for worker in range(4):
        children.append(launch(['-m','mem_lite.datasets.prepare_proprimem','worker','--output-dir',str(OUT),
                                '--model-path','/cache/cvpzhr/models/Qwen3.5-4B','--worker',str(worker)],f'worker-{worker}',worker))
    state('building',workers=[p.pid for p in children])
    while any(p.poll() is None for p in children):
        failed=[p.pid for p in children if p.poll() not in (None,0)]
        if failed:raise RuntimeError(f'数据构建失败，需复核：{failed}')
        if shutil.disk_usage(OUT).free<10*2**30:raise RuntimeError('磁盘剩余不足10 GiB；停止本次数据构建')
        time.sleep(10)
    if any(p.returncode for p in children):raise RuntimeError('worker 非正常结束')
    state('verifying')
    p=launch(['-m','mem_lite.datasets.prepare_proprimem','finalize','--output-dir',str(OUT)],'finalize','')
    children=[p]
    if p.wait()!=0:raise RuntimeError('最终校验失败，检查 finalize.log')
    state('completed',report=str(OUT/'report.json'),training_started=False)
except Exception as exc:
    for p in children:
        if p.poll() is None:p.terminate()
    for p in children:
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:p.kill();p.wait()
    state('needs_review',error=f'{type(exc).__name__}: {exc}',training_started=False)
    raise
