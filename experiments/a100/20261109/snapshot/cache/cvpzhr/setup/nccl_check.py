import json
import os
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist

rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl',timeout=timedelta(seconds=120),device_id=torch.device('cuda',rank))
try:
    x=torch.full((1024*1024,),rank+1.,device='cuda')
    dist.all_reduce(x)
    assert x.eq(10).all()
    a=torch.randn(512,512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=(a@a.T).float().square().mean()
    y.backward()
    assert torch.isfinite(y) and torch.isfinite(a.grad).all()
    result={'rank':rank,'name':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'bf16':torch.cuda.is_bf16_supported(),'all_reduce_sum':x[0].item()}
    results=[None]*4
    dist.all_gather_object(results,result)
    if rank==0:
        report={'status':'PASS','gpus':results,'torch':torch.__version__,'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version(),'parameter_updates':0}
        Path('/cache/cvpzhr/setup/nccl-check.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
finally: dist.destroy_process_group()
