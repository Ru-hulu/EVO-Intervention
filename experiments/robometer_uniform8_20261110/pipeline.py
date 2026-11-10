"""Strictly serial: cache all tasks, then smoke/train/validate/plot each on GPUs 2,3.

Never restarts a failed training from scratch. A nonempty incomplete output stops
the queue for an explicit checkpoint recovery audit. Existing results are kept.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def alive(pid):
    if not pid:return False
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class Pipeline:
    def __init__(self):
        self.policy=read(ROOT/'launch-policy.json')
        self.task=None;self.phase='initializing';self.child=None;self.completed=[]
        self.started=time.time()
        self.env=dict(os.environ,PYTHONPATH=str(ROOT/'runtime')+':'+self.policy['upstream'],
                      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      PYTHONNOUSERSITE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        self.python=self.policy['python']

    def status(self,phase,**extra):
        self.phase=phase
        row=dict(phase=phase,task=self.task,task_order=self.policy['task_order'],
                 completed_tasks=self.completed,gpus=self.policy['gpus'],pid=os.getpid(),
                 child_pid=self.child.pid if self.child and self.child.poll() is None else None,
                 started_at=self.started,updated_at=time.time(),**extra)
        write(ROOT/'status.json',row)
        if self.task:write(ROOT/self.task/'status.json',row)

    def run(self,cmd,phase,log,gpu=False,output=None):
        env=dict(self.env,CUDA_VISIBLE_DEVICES=','.join(map(str,self.policy['gpus'])) if gpu else '')
        log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            stream.write('\n'+json.dumps(dict(started=time.time(),command=cmd))+'\n');stream.flush()
            self.child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            self.status(phase,log=str(log))
            while self.child.poll() is None:
                current_phase=phase
                if phase=='training' and output and (Path(output)/'lifecycle.jsonl').exists():
                    events=(Path(output)/'lifecycle.jsonl').read_text().splitlines()
                    if events:
                        last=json.loads(events[-1])
                        if last['event']=='validation_started':current_phase='validating'
                self.status(current_phase,log=str(log))
                time.sleep(10)
            code=self.child.returncode;self.child=None
            if code:raise RuntimeError(f'{phase} failed (exit {code}); inspect {log}')

    def wait_gpus(self):
        free_checks=0
        while free_checks<2:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            use={int(r.split(',')[0]):[int(x.strip()) for x in r.split(',')[1:]] for r in rows.splitlines()}
            free=all(use[g][0]<200 and use[g][1]<5 for g in self.policy['gpus'])
            free_checks=free_checks+1 if free else 0
            self.status('waiting_for_gpus',gpu_usage=use)
            time.sleep(10)

    def main(self,prepare_only):
        checksum=read(ROOT/'code-manifest.json')
        for path,digest in checksum['files'].items():assert sha(ROOT/path)==digest,f'Code changed: {path}'
        self.run([self.python,str(ROOT/'check_sampling.py')],'checking_sampling',ROOT/'sampling.log')
        for task in self.policy['task_order']:
            self.task=task;config=read(self.policy['tasks'][task])
            self.run([self.python,str(ROOT/'build_cache.py'),'--config',self.policy['tasks'][task],'--workers','8'],
                     'building_cache',ROOT/task/'cache.log')
            receipt=read(Path(config['train_data']).parent/'cache-ready.json')
            assert receipt['status']=='passed' and receipt['all_cached_online_inputs_equal']
        self.task=None
        if prepare_only:
            self.status('caches_ready');return
        assert read(ROOT/'plotting-verification.json')['status']=='passed', 'Plotting/data-source preflight required'
        for task in self.policy['task_order']:
            self.task=task;config_path=self.policy['tasks'][task];config=read(config_path)
            output=Path(config['output_dir'])
            torchrun=[self.python,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                      str(ROOT/'train_official.py'),'--config',config_path]
            if (output/'report.json').exists():
                # Training may have finished before a plotting interruption; verify it again.
                assert read(output/'report.json')['status']=='completed'
            else:
                assert not output.exists() or not any(output.iterdir()), 'Incomplete training output: checkpoint recovery audit required'
                smoke=ROOT/task/'smoke'
                if all((smoke/f'smoke-verification-rank{r}.json').exists() for r in [0,1]):
                    for r in [0,1]:assert read(smoke/f'smoke-verification-rank{r}.json')['status']=='passed'
                else:
                    assert not (smoke/'output').exists() or not any((smoke/'output').iterdir()), 'Incomplete smoke trial requires inspection'
                    self.wait_gpus()
                    self.run(torchrun+['--smoke'],'smoke',ROOT/task/'smoke.log',gpu=True)
                    for r in [0,1]:assert read(smoke/f'smoke-verification-rank{r}.json')['status']=='passed'
                self.wait_gpus()
                self.run(torchrun,'training',ROOT/task/'train.log',gpu=True,output=output)
            self.run([self.python,str(ROOT/'verify_and_plot.py'),'--config',config_path],
                     'verifying_and_plotting',ROOT/task/'verification.log')
            manifest=read(output/'evaluation/curves/manifest.json')
            assert manifest['status']=='passed' and manifest['episodes']==config['expected_validation_episodes']
            assert manifest['samples']==config['expected_validation_samples']
            for name,digest in manifest['files'].items():assert sha(output/'evaluation/curves'/name)==digest
            self.completed.append(task);self.status('task_completed',output=str(output))
        results={task:read(Path(read(p)['output_dir'])/'evaluation/verified-metrics.json') for task,p in self.policy['tasks'].items()}
        write(ROOT/'comparison.json',results);self.task=None;self.status('completed')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    with (ROOT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'status.json').exists():
            old=read(ROOT/'status.json')
            assert not alive(old.get('child_pid')), 'Previous child is still alive; refusing duplicate work'
        pipeline=Pipeline()
        try:pipeline.main(a.prepare_only)
        except BaseException as exc:
            pipeline.status('failed',error=repr(exc),traceback=traceback.format_exc())
            raise


if __name__=='__main__':main()
