"""Short preflight, fresh three-epoch training and final validation. No restart or shutdown."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

RUN=Path(__file__).resolve().parent
SETTINGS=json.loads((RUN/'settings.json').read_text())
CODE=Path(SETTINGS['code']);GPUS=SETTINGS['gpus'];WORLD=len(GPUS)
CONFIG=RUN/'config.json'
STATE={'pid':os.getpid(),'created_at':time.time(),'training_authorized':True,'shutdown_authorized':False,
       'physical_gpus':GPUS,'world_size':WORLD,'code':str(CODE),'checkpoint_format':'mem-lite-streamforest-crossattn-v2'}

def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def status(phase,**values):
    STATE.update(phase=phase,updated_at=time.time(),**values);write_json(RUN/'status.json',STATE)

def gpu_sample():
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    gpus=[]
    for line in query.strip().splitlines():
        index,uuid,util,used,total=line.split(', ')
        gpus.append({'index':int(index),'uuid':uuid,'util':int(util),'memory_mib':int(used),'total_mib':int(total)})
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    processes=[line.split(', ') for line in query.strip().splitlines() if line]
    return {'time':time.time(),'gpus':gpus,'processes':processes}

def idle():
    sample=gpu_sample();used={row[0] for row in sample['processes']}
    return all(gpu['uuid'] not in used and gpu['util']<=5 and gpu['memory_mib']<1024
               for gpu in sample['gpus'] if gpu['index'] in GPUS)

def await_idle(stage):
    status('waiting_for_idle_gpu',next_stage=stage)
    while True:
        if idle():
            time.sleep(15)
            if idle():return
        time.sleep(15)

def run_stage(stage,args,output):
    await_idle(stage)
    env={**os.environ,'PYTHONPATH':str(CODE),'CUDA_VISIBLE_DEVICES':','.join(map(str,GPUS)),
         'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1',
         'WANDB_DISABLED':'true','PYTHONDONTWRITEBYTECODE':'1','NCCL_DEBUG':'WARN'}
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={WORLD}',*args]
    with (RUN/f'{stage}.log').open('w') as log:
        child=subprocess.Popen(command,cwd=CODE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    began=time.time();status(stage,child_pid=child.pid,stage_started_at=began)
    if stage=='training':STATE['training_started_at']=began
    try:
        while child.poll() is None:
            free=shutil.disk_usage(output).free/2**30
            with (RUN/'gpu.jsonl').open('a') as stream:stream.write(json.dumps(gpu_sample())+'\n')
            status(stage,free_gib=free)
            if free<2:raise RuntimeError('Disk free below 2 GiB; stopping only this job')
            if stage=='preflight' and time.time()-began>900:raise RuntimeError('Preflight exceeded 15 minutes')
            time.sleep(10 if stage=='preflight' else 20)
        if child.returncode:raise RuntimeError(f'{stage} failed with exit {child.returncode}; see {stage}.log')
        return time.time()-began
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def main():
    with (RUN/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'started.json').exists():raise RuntimeError('Already started; automatic restart prohibited')
        config=json.loads(CONFIG.read_text());output=Path(config['output_dir'])
        assert config['epochs']==3 and config['save_every_epochs']==3 and config['keep_only_latest']
        assert config['save_every_steps'] is None and config['max_steps'] is None and config['batch_size']==4
        assert not output.exists() or not any(output.iterdir())
        output.mkdir(parents=True,exist_ok=True)
        expected_steps=3*math.ceil(math.ceil(126441/WORLD)/4)
        STATE.update(output=str(output),epochs=3,expected_steps=expected_steps,steps_per_epoch=expected_steps//3,
                     batch_per_gpu=4,effective_batch=4*WORLD,train_samples=126441,validation_samples=32164)
        digest=hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        write_json(RUN/'started.json',{'time':time.time(),'config_sha256':digest,'expected_steps':expected_steps})
        try:
            preflight=RUN/'preflight';preflight.mkdir()
            probe={**config,'output_dir':str(preflight),'epochs':1,'max_steps':6,'log_every_steps':1,'save_every_epochs':None}
            write_json(RUN/'preflight.json',probe)
            seconds=run_stage('preflight',['-m','mem_lite.train_stage3','--config',str(RUN/'preflight.json')],preflight)
            import torch
            checkpoint=torch.load(preflight/'last.pt',map_location='cpu',weights_only=True)
            assert checkpoint['format']=='mem-lite-streamforest-crossattn-v2' and checkpoint['step']==6
            gates={k:float(v) for k,v in checkpoint['model']['memory_adapters'].items() if k.endswith('.gate')}
            assert len(gates)==3 and all(math.isfinite(v) and v!=0 for v in gates.values())
            rows=[json.loads(line) for line in (preflight/'train.jsonl').read_text().splitlines()]
            assert len(rows)==6 and all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows)
            write_json(RUN/'preflight-result.json',{'status':'passed','updates':6,'world_size':WORLD,'seconds':seconds,
                       'gates':gates,'records':rows,'formal_training_will_restart_from_original_base':True})
            del checkpoint
            (preflight/'last.pt').unlink()  # This run's temporary smoke-test checkpoint only.
            training_seconds=run_stage('training',['-m','mem_lite.train_stage3','--config',str(CONFIG)],output)
            checkpoint=torch.load(output/'last.pt',map_location='cpu',weights_only=True)
            assert checkpoint['format']=='mem-lite-streamforest-crossattn-v2'
            assert checkpoint['epoch']==3 and checkpoint['step']==expected_steps and checkpoint['next_batch']==0
            assert sorted(p.name for p in output.glob('*.pt'))==['last.pt']
            del checkpoint
            validation_seconds=run_stage('validation',[str(RUN/'validate.py')],output)
            rows=[json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
            metrics=json.loads((output/'validation-03/metrics.json').read_text())
            assert metrics['num_samples']==32164
            report={'status':'completed','epochs':3,'steps':expected_steps,'world_size':WORLD,'physical_gpus':GPUS,
                    'batch_per_gpu':4,'effective_batch':4*WORLD,'checkpoint':str(output/'last.pt'),
                    'checkpoint_format':'mem-lite-streamforest-crossattn-v2','training_seconds':training_seconds,
                    'validation_wall_seconds':validation_seconds,'config_sha256':digest,
                    'training':{'first_logged_loss':rows[0]['loss'],'last_logged_loss':rows[-1]['loss'],
                    'mean_logged_step_loss':sum(r['loss'] for r in rows)/len(rows),'logged_steps':len(rows),
                    'gradients_all_finite':all(math.isfinite(r['grad_norm']) for r in rows),
                    'max_logged_grad_norm':max(r['grad_norm'] for r in rows),
                    'peak_allocated_gib':max(max(r['peak_allocated_gib_per_gpu']) for r in rows),
                    'peak_reserved_gib':max(max(r['peak_reserved_gib_per_gpu']) for r in rows)},'validation':metrics}
            baseline=json.loads(Path('/cache/cvpzhr/outputs/forest-proprimem-e3-20261031-0009/validation-03/metrics.json').read_text())
            report['prior_forest_comparison']={'baseline':baseline,
                    'mae_difference_pp':metrics['mae_pp']-baseline['mae_pp'],
                    'rmse_difference_pp':metrics['rmse_pp']-baseline['rmse_pp'],
                    'note':'Same data/PT and validation batch groups. Current global train batch 12; prior baseline 16.'}
            write_json(output/'report.json',report)
            status('completed',completed_at=time.time(),training_seconds=training_seconds,validation_seconds=validation_seconds)
        except Exception as exc:
            write_json(RUN/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'time':time.time()})
            status('failed',error=str(exc));raise

if __name__=='__main__':main()
