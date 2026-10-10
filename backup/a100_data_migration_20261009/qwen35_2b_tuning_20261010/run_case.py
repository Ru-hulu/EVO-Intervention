"""Run one bounded 8-GPU benchmark; never launches formal training."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path

BASE=Path('/data/evo-inter/verification/qwen35_2b_tuning_20261010')
DOWNLOAD=Path('/data/evo-inter/verification/qwen35_2b_20261010')
MODEL=Path('/data/evo-inter/models/Qwen3.5-2B')
p=argparse.ArgumentParser(__doc__)
p.add_argument('--name',required=True)
p.add_argument('--per-gpu-batch',type=int,required=True)
p.add_argument('--train-batches',type=int,default=32)
p.add_argument('--val-batches',type=int,default=2)
p.add_argument('--benchmark-epochs',type=int,default=1)
p.add_argument('--val-batch-size',type=int,default=16)
p.add_argument('--measure-input',action='store_true')
p.add_argument('--num-workers',type=int,default=4)
a=p.parse_args()
if not a.name.replace('-','').replace('_','').isalnum() or not 1<=a.per_gpu_batch<=16:p.error('Invalid case name or batch')
if not 1<=a.train_batches<=128 or not 1<=a.val_batches<=32 or not 1<=a.benchmark_epochs<=2:p.error('Only short benchmarks are allowed')
BASE.mkdir(exist_ok=True)
lock=(BASE/'active.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
result=json.loads((DOWNLOAD/'download_result.json').read_text())
if result.get('status')!='complete':raise RuntimeError('Model download has not completed successfully')
manifest=json.loads((DOWNLOAD/'modelscope_manifest.json').read_text())['Data']['Files']
for f in manifest:
 if f['Type']=='blob' and (MODEL/f['Path']).stat().st_size!=f['Size']:raise RuntimeError('Downloaded file size mismatch')
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
if busy:raise RuntimeError('GPUs have existing compute processes; refusing to launch another job')
r=BASE/a.name;r.mkdir(exist_ok=False)
command=['/data/evo-inter/.venv/bin/python','-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8','-m','qwen_training.benchmark_distributed',
 '--benchmark-epochs',str(a.benchmark_epochs),'--train-batches',str(a.train_batches),'--val-batches',str(a.val_batches),
 '--model-path',str(MODEL),'--index-files','/root/dataset/RW-RL-Dataset/piperx_insert_copper_screw/samples/samples.jsonl','/root/dataset/piperx-bolt-intentional-drop-reinsert-22ep-raw/samples/samples.jsonl',
 '--output-dir',str(r/'output'),'--global-batch-size',str(a.per_gpu_batch*8),'--val-batch-size',str(a.val_batch_size),
 '--num-workers',str(a.num_workers),'--prefetch-factor','2','--gpu-image-processing','--require-fast-kernels','--max-pixels','262144',
 '--lr','0.0001','--lora-rank','16','--lora-alpha','32','--lora-dropout','0.05','--text-loss-weight','0.7','--progress-loss-weight','0.3']
if a.measure_input:
 module_index=command.index('qwen_training.benchmark_distributed')
 command[module_index-1:module_index+1]=[str(BASE/'benchmark_input_timing.py')]
env=os.environ.copy()
env.update(PYTHONPATH='/data/evo-inter/code',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
# Use NCCL defaults; forced PHB transport was slower in this host's isolated test.
for key in ['NCCL_P2P_LEVEL','NCCL_P2P_DISABLE']:env.pop(key,None)
started=time.time()
with (r/'run.log').open('x') as out:
 child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,cwd='/data/evo-inter/code',env=env,start_new_session=True)
 (r/'pid').write_text(str(child.pid)+'\n')
 launch={'name':a.name,'driver_pid':os.getpid(),'torchrun_pid':child.pid,'started_unix':started,'command':command,'parameters':vars(a),'formal_training':False}
 (r/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 (BASE/'active_case.json').write_text(json.dumps(launch,indent=2)+'\n')
 print(json.dumps({'event':'benchmark_started',**launch}),flush=True)
 with (r/'resources.jsonl').open('x',buffering=1) as log:
  while child.poll() is None:
   mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
   available=int(mem['MemAvailable'].split()[0])/2**20
   gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
   rows=[]
   if gpu.returncode==0:
    for line in gpu.stdout.splitlines():
     fields=[x.strip() for x in line.split(',')]
     rows.append({'index':int(fields[0]),'used_mib':float(fields[1]),'total_mib':float(fields[2]),'utilization':float(fields[3])})
   log.write(json.dumps({'timestamp':time.time(),'host_available_gib':available,'gpus':rows})+'\n')
   if available<100 or time.time()-started>1800:
    (r/'abort_reason.txt').write_text('Host RAM reserve below 100 GiB or benchmark exceeded 30 minutes')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
    break
   time.sleep(5)
 rc=child.wait()
 (r/'exit_code.txt').write_text(str(rc)+'\n')
metrics_path=r/'output/metrics.jsonl'
metrics=[json.loads(x) for x in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
# Exclude the first four steps of each shortened epoch (loader start, lazy initialization).
train=[x for x in metrics if x['kind']=='train' and ((x['step']-1)%a.train_batches)>=4]
rank_paths=sorted((r/'output').glob('rank-*.json'))
ranks=[json.loads(f.read_text()) for f in rank_paths]
resources=[json.loads(x) for x in (r/'resources.jsonl').read_text().splitlines()]
gpu_rows=[g for x in resources for g in x['gpus']]
report={'case':a.name,'exit_code':rc,'per_gpu_batch':a.per_gpu_batch,'global_batch':a.per_gpu_batch*8,'measured_steps_after_warmup':len(train),
 'steady_samples_per_second':sum(x['samples'] for x in train)/sum(x['samples']/x['samples_per_second'] for x in train) if train else None,
 'peak_allocated_mib':max((x['peak_allocated_mib'] for x in ranks),default=None),'peak_reserved_mib':max((x['peak_reserved_mib'] for x in ranks),default=None),
 'sampled_peak_device_used_mib':max((x['used_mib'] for x in gpu_rows),default=None),'sampled_min_device_free_mib':min((x['total_mib']-x['used_mib'] for x in gpu_rows),default=None),
 'minimum_host_available_gib':min((x['host_available_gib'] for x in resources),default=None),'rank_results':ranks,
 'validation':[x for x in metrics if x['kind']=='validation'],'phases':[x for x in metrics if x['kind']=='phase'],
 'completed_unix':time.time(),'formal_training':False,'note':'Real-data bounded benchmark. CUDA allocator peaks span training and validation. Device memory sampled every five seconds can miss transient peaks. Throughput excludes warmup and validation; best batch is only among tested settings.'}
(r/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
(BASE/'active_case.json').write_text(json.dumps({**launch,'exit_code':rc,'completed_unix':time.time()},indent=2)+'\n')
print(json.dumps(report),flush=True)
sys.exit(rc)
