"""Start evaluation only after the authorized Robometer epoch and clean validation finish."""
import fcntl, os, signal, subprocess, sys, time, traceback
from common import *
def training_ready(status):
    if status.get('phase')=='failed':raise RuntimeError('Robometer training failed; evaluation was not started')
    return status.get('phase')=='completed' and status.get('step')==9967 and abs(status.get('epoch',-1)-1)<1e-6
def gpu_apps():
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    return [x.strip() for x in result.splitlines() if x.strip()]
def environment(model,rank):
    code=F_CODE if model in ('cache','forest') else R_CODE
    paths=[str(ROOT),str(code)]
    if model=='robometer':paths.append(str(R_CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'))
    return dict(os.environ,PYTHONPATH=':'.join(paths),CUDA_VISIBLE_DEVICES=str(rank),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONNOUSERSITE='1')
def main():
    lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    children=[]
    started=now()
    def state(phase,**kw):
        write(ROOT/'status.json',dict(phase=phase,pid=os.getpid(),started_at=started,updated_at=now(),
             metrics=str(OUT/'metrics.json'),**kw))
    def stop():
        for proc in children:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
        for proc in children:
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    def interrupted(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def wait_free(phase):
        while True:
            apps=gpu_apps()
            if not apps:return
            state('waiting_for_gpu_release',next_phase=phase,compute_pids=apps)
            time.sleep(30)
    def run_phase(kind):
        nonlocal children
        wait_free(kind)
        state(kind)
        for rank in range(4):
            python=R_PY if kind=='robometer' else F_PY
            with (ROOT/f'{kind}-{rank}.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([python,'-u',str(ROOT/'worker.py'),kind,'--rank',str(rank)],
                    cwd=ROOT,env=environment(kind,rank),stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
            children.append(child)
        while any(c.poll() is None for c in children):
            if any(c.poll() not in (None,0) for c in children):raise RuntimeError(f'{kind} worker failed; inspect {kind}-*.log')
            progress={str(r):read(ROOT/f'{kind}-{r}.json') for r in range(4) if (ROOT/f'{kind}-{r}.json').exists()}
            state(kind,workers=progress)
            time.sleep(15)
        assert all(c.returncode==0 for c in children)
        children=[]
    try:
        assert read(ROOT/'cpu-tests.json')['status']=='passed'
        assert read(ROOT/'input-checks.json')['status']=='passed'
        p=plan();check_inputs(p)
        hashes=read(ROOT/'implementation-sha256.json')
        assert all(sha(x)==digest for x,digest in hashes.items())
        while True:
            status=read(R_SETUP/'training-status.json')
            if training_ready(status):break
            pid=status.get('pid')
            if pid and not Path(f'/proc/{pid}').exists():
                raise RuntimeError('Training controller is absent without a completed status')
            state('waiting_for_training',training_phase=status.get('phase'),training_step=status.get('step'),
                  training_epoch=status.get('epoch'),expected_step=9967)
            time.sleep(30)
        report=read(R_OUT/'report.json')
        assert report['status']=='completed' and report['epochs']==1 and report['steps']==9967
        receipt=read(R_CKPT/'complete.json')
        assert receipt['complete'] and receipt['step']==9967 and abs(receipt['epoch']-1)<1e-6
        assert read(R_CKPT/'validation.json')['eval_samples']==66891
        check_inputs(p)
        write(OUT/'checkpoints.json',dict(forest=dict(path=str(F_CKPT),epoch=1,step=19934,sha256=sha(F_CKPT)),
              robometer=dict(path=str(R_CKPT),epoch=1,step=9967,files={n:sha(R_CKPT/n) for n in
              ['adapter_config.json','adapter_model.safetensors','progress_head.safetensors','run_config.json','trainer_state.json']})))
        if not (CACHE/'report.json').exists():
            run_phase('cache')
            receipts=[read(ROOT/'receipts'/(e['episode_id']+'.json')) for e in p['episodes']]
            assert sum(r['samples'] for r in receipts)==p['augmented_samples']
            assert all(r['inserted_pt_fixed_to_anchor'] for r in receipts)
            write(CACHE/'validation.sample_files.json',[e['output_dir'] for e in p['episodes']])
            write(CACHE/'report.json',dict(status='completed',episodes=417,samples=p['augmented_samples'],
                                          plan_sha256=p['plan_sha256'],kinds=p['kinds']))
        else:assert read(CACHE/'report.json')['plan_sha256']==p['plan_sha256']
        run_phase('forest')
        run_phase('robometer')
        state('aggregating')
        with (ROOT/'aggregation.log').open('ab') as log:
            subprocess.run([sys.executable,str(ROOT/'aggregate.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(OUT/'metrics.json')['status']=='completed'
        state('completed',comparison=str(OUT/'comparison.md'))
    except BaseException as exc:
        stop();state('failed',error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
