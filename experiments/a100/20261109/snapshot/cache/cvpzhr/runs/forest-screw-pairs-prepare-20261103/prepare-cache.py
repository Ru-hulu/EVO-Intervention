import json, os, pathlib, subprocess, sys, time, traceback
run = pathlib.Path('/cache/cvpzhr/runs/forest-screw-pairs-prepare-20261103')
code = '/cache/cvpzhr/code/forest-screw-pairs-20261103'
prepared = '/cache/EVO-Intervention/datasets/insert-screw-recover-pairs-20261103'
cache = '/cache/cvpzhr/datasets/forest-screw-pairs-5fps-20261103'
python = '/cache/cvpzhr/envs/forest/bin/python'
reference = '/cache/cvpzhr/outputs/forest-crossattn-linear-e3-restart-20261102-2352/config.json'
def status(phase, **extra):
    value = dict(phase=phase, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), dataset=prepared, cache=cache, code=code, training_started=False, **extra)
    (run/'status.json.tmp').write_text(json.dumps(value,indent=2)+'\n')
    (run/'status.json.tmp').replace(run/'status.json')
    print(json.dumps(value), flush=True)
try:
    status('waiting_for_clips')
    while not pathlib.Path(prepared,'clips-report.json').is_file():
        pid = int((run/'clips.pid').read_text())
        os.kill(pid,0)
        time.sleep(5)
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used = {int(a):int(b) for a,b in (line.split(',') for line in gpu.splitlines())}
    if any(used[i]>1024 for i in (1,2,3)):
        raise RuntimeError('A selected cache GPU is no longer idle: '+gpu)
    jobs = []
    for worker, device in enumerate((1,2,3)):
        log = (run/f'cache-{worker}.log').open('ab')
        command = [python,'-u','-m','mem_lite.datasets.prepare_recovery_pairs','cache','--prepared-dir',prepared,'--cache-dir',cache,'--model-path','/cache/cvpzhr/models/Qwen3.5-4B','--worker',str(worker),'--workers','3']
        proc = subprocess.Popen(command,cwd=code,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(device),'OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'},stdout=log,stderr=subprocess.STDOUT)
        log.close()
        (run/f'cache-{worker}.pid').write_text(str(proc.pid)+'\n')
        jobs.append(proc)
    status('caching', worker_pids=[p.pid for p in jobs])
    exits = [p.wait() for p in jobs]
    if any(exits): raise RuntimeError('Cache worker exit codes: '+str(exits))
    status('validating')
    with (run/'cache-validation.log').open('ab') as log:
        subprocess.run([python,'-u','-m','mem_lite.datasets.validate_recovery_pairs','--prepared-dir',prepared,'--cache-dir',cache,'--reference-config',reference],cwd=code,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},stdout=log,stderr=subprocess.STDOUT,check=True)
    status('cache_verified_preflight_pending')
except Exception:
    status('preparation_failed',error=traceback.format_exc())
    raise
