import os,sys,torch
from pathlib import Path
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0,str(Path(__file__).parent/'pencil/code'))
from mem_lite.train_stage3 import normalize_gradients
torch.set_num_threads(1)
dist.init_process_group('gloo')
rank=dist.get_rank()
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.main=torch.nn.Linear(3,1,dtype=torch.float64);self.history=torch.nn.Linear(2,1,dtype=torch.float64)
 def forward(self,x,h,use):return self.main(x)+(self.history(h) if use else 0)
torch.manual_seed(42)
model=Toy();gold=Toy();gold.load_state_dict(model.state_dict());wrapped=DDP(model,find_unused_parameters=True)
for flags in [[(True,True),(True,False),(False,False)],[(False,True),(True,True),(True,False)]]:
 model.zero_grad(set_to_none=True);gold.zero_grad(set_to_none=True)
 all_inputs=[]
 for micro,uses in enumerate(flags):
  n=3 if micro<2 else 1
  for r in range(2):
   gen=torch.Generator().manual_seed(100+micro*10+r)
   x=torch.randn(n,3,generator=gen,dtype=torch.float64);h=torch.randn(n,2,generator=gen,dtype=torch.float64);y=torch.randn(n,1,generator=gen,dtype=torch.float64)
   all_inputs.append((x,h,y,uses[r]))
   if r==rank:
    with wrapped.no_sync() if micro<2 else nullcontext():((wrapped(x,h,uses[r])-y)**2).sum().backward()
 total=sum(len(x) for x,_,_,_ in all_inputs)
 normalize_gradients(list(model.parameters()),total,2)
 for x,h,y,use in all_inputs:((gold(x,h,use)-y)**2).sum().div(total).backward()
 for (name,p),q in zip(model.named_parameters(),gold.parameters()):assert torch.allclose(p.grad,q.grad,rtol=1e-12,atol=1e-12),(rank,name,p.grad,q.grad)
if rank==0:print('PASS: DDP accumulation matches global-batch gradients with missing history on final microbatches',flush=True)
dist.destroy_process_group()
