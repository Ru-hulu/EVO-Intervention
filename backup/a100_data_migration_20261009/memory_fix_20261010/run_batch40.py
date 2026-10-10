import json,os,signal,subprocess,time
from pathlib import Path
b=Path('/data/evo-inter/verification/memory_fix_20261010')
r=b/'batch40-test';r.mkdir(exist_ok=False)
command="source /data/evo-inter/env.sh; export PYTHONPATH="+str(b/'candidate')+"; export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1; cd "+str(b/'candidate')+"; exec python -u -m torch.distributed.run --standalone --nproc_per_node=8 -m qwen_training.benchmark_distributed --benchmark-epochs 1 --train-batches 16 --val-batches 1 --model-path /data/evo-inter/models/Qwen3.5-4B --index-files /root/dataset/RW-RL-Dataset/piperx_insert_copper_screw/samples/samples.jsonl /root/dataset/piperx-bolt-intentional-drop-reinsert-22ep-raw/samples/samples.jsonl --output-dir "+str(r/'output')+" --global-batch-size 40 --val-batch-size 16 --num-workers 4 --prefetch-factor 2 --gpu-image-processing --require-fast-kernels"
with (r/'run.log').open('w') as out:
 p=subprocess.Popen(['bash','-c',command],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 (r/'pid').write_text(str(p.pid))
 start=time.time()
 with (r/'memory.jsonl').open('w',buffering=1) as log:
  while p.poll() is None:
   table=[]
   for line in subprocess.check_output(['ps','-e','-o','pid=,ppid=,rss=,nlwp=,comm='],text=True).splitlines():
    parts=line.split(None,4)
    table.append(dict(pid=int(parts[0]),ppid=int(parts[1]),rss_kib=int(parts[2]),threads=int(parts[3]),name=parts[4]))
   descendants={p.pid}
   while True:
    newer=descendants|{x['pid'] for x in table if x['ppid'] in descendants}
    if newer==descendants: break
    descendants=newer
   own=[x for x in table if x['pid'] in descendants]
   mem=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
   available=int(mem['MemAvailable'].split()[0])/2**20
   row={'timestamp':time.time(),'rss_sum_gib':sum(x['rss_kib'] for x in own)/2**20,'host_available_gib':available,'workers':[x for x in own if x['name']=='pt_data_worker']}
   log.write(json.dumps(row)+'\n')
   if available<100 or time.time()-start>1800:
    (r/'abort_reason.txt').write_text('Memory reserve under 100 GiB or 30 minute timeout')
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
    break
   time.sleep(5)
 rc=p.wait()
 (r/'exit_code.txt').write_text(str(rc)+'\n')
 print('TRANSITION_TEST_EXIT',rc,flush=True)
