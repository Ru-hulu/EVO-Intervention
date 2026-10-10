"""Communication-only microbenchmark. No model, dataset or optimizer is loaded."""
import json, os, statistics, time
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

b=Path('/data/evo-inter/verification/communication_20261010')
rank=int(os.environ['RANK']); device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
torch.cuda.set_device(device)
dist.init_process_group('nccl',device_id=device,timeout=timedelta(seconds=120))
world=dist.get_world_size()
shapes=json.loads((b/'parameter_sizes.json').read_text())
out=Path(os.environ.get('COMM_BENCH_OUTPUT',str(b)))
out.mkdir(exist_ok=True)
numel=sum(v['numel'] for v in shapes.values())
rows=[]
# Bucket capacity applies to original FP32 gradient elements in both cases.
cases=[('fp32',torch.float32,8),('fp32',torch.float32,25),('fp32',torch.float32,50),('bf16',torch.bfloat16,8),('bf16',torch.bfloat16,25),('bf16',torch.bfloat16,50),('scalar2',torch.float64,None)]
for name,dtype,cap in cases:
 count=2 if cap is None else numel
 chunk=count if cap is None else int(cap*2**20/4)
 tensors=[torch.full((min(chunk,count-start),),rank+1,dtype=dtype,device=device) for start in range(0,count,chunk)]
 def reduce():
  pending=[dist.all_reduce(t,async_op=True) for t in tensors]
  for work in pending: work.wait()
 reduce();torch.cuda.synchronize()
 assert all(torch.all(t.eq(world*(world+1)/2)).item() for t in tensors)
 for t in tensors:t.zero_()
 for _ in range(5): reduce()
 torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
 times=[]
 for _ in range(30):
  start=time.perf_counter();reduce();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
 row={'rank':rank,'dtype':name,'bucket_cap_fp32_mib':cap,'num_buckets':len(tensors),'payload_bytes':sum(t.numel()*t.element_size() for t in tensors),'median_ms':statistics.median(times),'mean_ms':statistics.mean(times),'min_ms':min(times),'max_ms':max(times),'times_ms':times}
 rows.append(row)
 del tensors
 gathered=[None]*world if rank==0 else None
 dist.gather_object(row,gathered,dst=0)
 if rank==0:
  print(json.dumps({'dtype':name,'bucket_cap_fp32_mib':cap,'payload_mib':row['payload_bytes']/2**20,'slowest_rank_median_ms':max(x['median_ms'] for x in gathered)}),flush=True)
all_rows=[None]*world if rank==0 else None
dist.gather_object(rows,all_rows,dst=0)
if rank==0:
 report={'world_size':world,'torch_version':torch.__version__,'nccl_version':torch.cuda.nccl.version(),'parameters':numel,'warmup_iterations':5,'measured_iterations':30,'ranks':all_rows,'limitations':['Synthetic communication-only test; all buckets are ready together. Actual DDP bucket membership and gradient readiness differ.','No simultaneous backward kernels, no communication/computation overlap measured.','BF16 case measures reduced payload only; excludes compression/decompression and numerical impact.','Transport capacity is not the exposed communication overhead in real training.']}
 (out/'collectives.json').write_text(json.dumps(report,indent=2)+'\n')
dist.destroy_process_group()
