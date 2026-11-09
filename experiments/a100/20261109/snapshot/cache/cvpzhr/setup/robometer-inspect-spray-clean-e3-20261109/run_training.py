"""Detached two-GPU fine-tuning controller; cache and checkpoint must pass first."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SETUP=Path('/cache/cvpzhr/setup/robometer-inspect-spray-clean-e3-20261109')
CONFIG=SETUP/'training.config.json'
PY='/cache/cvpzhr/envs/robometer/bin/python'
THIRD='/cache/cvpzhr/code/robometer-current3-w8-e1-20261104/third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'

def main():
    with (SETUP/'training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads(CONFIG.read_text());output=Path(config['output_dir'])
        state={'pid':os.getpid(),'started_at':time.time(),'output_dir':str(output),
               'task':'inspect-spray-bottles','epochs':3,'augmentation':'none','training_started':False}
        def update(**values):
            state.update(values,updated_at=time.time())
            p=SETUP/'training-status.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(SETUP/'training-status.json')
            print(json.dumps(state),flush=True)
        try:
            update(phase='waiting_for_clean_cache')
            while True:
                cache=json.loads((SETUP/'cache-controller.json').read_text())
                if cache['phase']=='failed':raise RuntimeError('Clean cache generation failed')
                if cache['phase']=='completed':break
                time.sleep(10)
            root=Path(config['train_data']).parent
            ready=json.loads((root/'cache-ready.json').read_text())
            audit=json.loads((root/'label-audit.json').read_text())
            assert ready['status']=='ready' and ready['augmentation']=='none' and audit['status']=='passed'
            assert not output.exists() or not any(output.iterdir()), 'Refusing to reuse a nonempty output directory'
            checkpoint=Path(config['initial_checkpoint'])
            saved=json.loads((checkpoint/'trainer_state.json').read_text());assert abs(saved['epoch']-1)<1e-6
            hashes={}
            for name in ['adapter_config.json','adapter_model.safetensors','progress_head.safetensors']:
                p=checkpoint/name;assert p.is_file();hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [CONFIG,*SETUP.glob('*.py'),*(SETUP/'code').rglob('*.py')]:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            (SETUP/'run-input-sha256.json').write_text(json.dumps(hashes,indent=2))
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
            occupied=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
            available=[]
            for line in raw.splitlines():
                index,uuid,memory=[x.strip() for x in line.split(',')]
                if uuid not in occupied and int(memory)<1024:available.append(int(index))
            assert len(available)>=2, 'Fewer than two idle GPUs; no existing job was interrupted'
            gpus=available[:2]
            env={**os.environ,'PYTHONPATH':str(SETUP/'code')+':'+THIRD,'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),
                 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
                 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONNOUSERSITE':'1','NCCL_DEBUG':'WARN'}
            command=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(SETUP/'train_from_checkpoint.py'),'--config',str(CONFIG)]
            with (SETUP/'train.log').open('w') as log:
                child=subprocess.Popen(command,cwd=SETUP,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                update(phase='training',training_started=True,launcher_pid=child.pid,gpu_ids=gpus,command=command)
                code=child.wait()
            assert code==0, f'Training/evaluation exited with code {code}; see train.log'
            report=json.loads((output/'report.json').read_text());assert report['status']=='completed'
            update(phase='completed',returncode=0,final_checkpoint=report['final_checkpoint'],metrics=report['evaluation'],elapsed_seconds=time.time()-state['started_at'])
        except BaseException as exc:
            update(phase='failed',error=repr(exc),elapsed_seconds=time.time()-state['started_at']);raise

if __name__=='__main__':main()
