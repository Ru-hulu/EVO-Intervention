"""Four-GPU forward/backward benchmark. No optimizer steps or saved model."""
import argparse,json,os,time,statistics
from pathlib import Path
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from robometer_training.data import ProgressDataset,ProgressCollator,_episode_cache
from robometer_training.model import load_model,load_processor,attach_lora,current_logits,progress_loss


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--batch',type=int,required=True);p.add_argument('--checkpointing',action='store_true');p.add_argument('--output',required=True);args=p.parse_args()
 rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(local);torch.set_num_threads(1)
 dist.init_process_group('nccl',timeout=timedelta(minutes=8),device_id=torch.device('cuda',local))
 c=json.loads(Path(args.config).read_text());processor=load_processor(c['model_path'],min_pixels=4096,max_pixels=65536)
 model,_=load_model(c['model_path'],processor);attach_lora(model,**c['lora']);model.cuda().train();model.model.visual.eval()
 if args.checkpointing:model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
 model=DDP(model,device_ids=[local],find_unused_parameters=False)
 parameters=[p for p in model.parameters() if p.requires_grad]
 # Reserve the same FP32 moment buffers as AdamW, without performing any update.
 adam_moments=[(torch.zeros_like(p,dtype=torch.float32),torch.zeros_like(p,dtype=torch.float32)) for p in parameters]
 data=ProgressDataset(c['train_data']);collator=ProgressCollator(processor)
 with open(c['benchmark_candidates']) as f: candidates=json.load(f)
 worst=max(candidates,key=lambda r:(r['tokens'],r['patches']))['row']
 batch=collator([worst]*args.batch);batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
 torch.cuda.reset_peak_memory_stats();losses=[];gradients=[]
 def step(batch):
  model.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   loss=progress_loss(current_logits(model,{k:v for k,v in batch.items() if k!='labels'}),batch['labels'])
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.0)
  if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('Non-finite loss/gradient')
  return float(loss.detach()),float(norm)
 for _ in range(2):step(batch)
 torch.cuda.synchronize();stress_alloc=torch.cuda.max_memory_allocated()/1024**3;stress_reserved=torch.cuda.max_memory_reserved()/1024**3
 del batch;model.zero_grad(set_to_none=True);torch.cuda.empty_cache();_episode_cache.cache_clear()
 sampler=DistributedSampler(data,world,rank,shuffle=True,seed=42,drop_last=True)
 loader=DataLoader(data,batch_size=args.batch,sampler=sampler,num_workers=4,persistent_workers=True,prefetch_factor=2,pin_memory=True,collate_fn=collator,multiprocessing_context='spawn',drop_last=True)
 iterator=iter(loader)
 for _ in range(3):
  batch=next(iterator);step({k:v.cuda(non_blocking=True) for k,v in batch.items()})
 torch.cuda.synchronize();dist.barrier();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();times=[];io=[]
 for _ in range(12):
  begin=time.perf_counter();batch=next(iterator);io.append(time.perf_counter()-begin)
  batch={k:v.cuda(non_blocking=True) for k,v in batch.items()};loss,grad=step(batch);torch.cuda.synchronize()
  times.append(time.perf_counter()-begin);losses.append(loss);gradients.append(grad)
 elapsed=time.perf_counter()-start
 stats=torch.tensor([elapsed,max(stress_alloc,torch.cuda.max_memory_allocated()/1024**3),max(stress_reserved,torch.cuda.max_memory_reserved()/1024**3),max(gradients),max(losses),sum(io)],device='cuda')
 dist.all_reduce(stats,op=dist.ReduceOp.MAX)
 if rank==0:
  result=dict(status='passed',per_gpu_batch=args.batch,global_batch=args.batch*world,world_size=world,gradient_checkpointing=args.checkpointing,workers_per_gpu=4,measured_steps=12,warmup_steps=3,stress_steps=2,seconds=float(stats[0]),samples_per_second=args.batch*world*12/float(stats[0]),peak_allocated_gib=float(stats[1]),peak_reserved_gib=float(stats[2]),max_grad_norm=float(stats[3]),max_loss=float(stats[4]),max_rank_data_wait_seconds=float(stats[5]),optimizer_steps=0,parameters_updated=False,adam_moments_reserved=True,stress_window_tokens=max(candidates,key=lambda r:(r['tokens'],r['patches']))['tokens'])
  Path(args.output).write_text(json.dumps(result,indent=2));print('BENCHMARK_RESULT',json.dumps(result),flush=True)
 del iterator,loader
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
