from pathlib import Path
import fcntl,json,os,shutil,subprocess,time,traceback
S=Path(__file__).resolve().parent
F=Path('/cache/cvpzhr/setup/forest-drawer-noaug-cache-20261109')
R=Path('/cache/cvpzhr/setup/robometer-drawer-clean-cache-20261109')
FD=Path('/cache/cvpzhr/datasets/forest-drawer-5fps-no-augmentation-20261109')
RD=Path('/cache/cvpzhr/datasets/robometer-drawer-5fps-h7-current3-clean-20261109')
def read(p):return json.loads(p.read_text())
def save(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
    with (S/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'pid':os.getpid(),'started_at':time.time(),'task':'drawer-inspection-close'}
        def update(phase,**kw):state.update(phase=phase,updated_at=time.time(),**kw);save(S/'status.json',state)
        def run(cmd,log,phase):
            update(phase,command=cmd,log=str(log))
            with log.open('a') as f:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'))
                update(phase,child_pid=child.pid);code=child.wait()
            update(phase,child_pid=None,returncode=code)
            assert code==0,(phase,code,str(log))
        try:
            update('preparing')
            assert not any(p.exists() for p in [F,R,FD,RD]), 'Existing drawer preparation requires recovery audit'
            F.mkdir();R.mkdir();(R/'source').mkdir()
            for src,dst in [('forest_cache.py',F/'prepare_cache.py'),('robometer_cache.py',R/'prepare_cache.py'),('robometer_run_cache.py',R/'run_cache.py'),('robometer_verify_labels.py',R/'verify_labels.py')]:shutil.copy2(S/src,dst)
            shutil.copytree('/cache/cvpzhr/setup/forest-robometer-spray-official-e3-20261109/robometer/code',R/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            run(['/cache/cvpzhr/envs/forest/bin/python','-u',str(F/'prepare_cache.py')],S/'forest-cache.log','verifying_forest_cache')
            assert read(FD/'cache-ready.json')['status']=='completed'
            for split in ['train','validation']:shutil.copy2(FD/f'{split}.sample_files.json',R/'source'/f'{split}.sample_files.json')
            run(['/cache/cvpzhr/envs/robometer/bin/python','-u',str(R/'run_cache.py')],S/'robometer-cache.log','building_robometer_cache')
            assert read(RD/'cache-ready.json')['status']=='ready'
            update('completed',finished_at=time.time(),forest_cache=str(FD),robometer_cache=str(RD))
        except BaseException as exc:
            update('failed',error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
