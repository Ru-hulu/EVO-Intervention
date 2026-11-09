"""Independent, non-overlapping four-GPU evaluation of the final adapter."""
import argparse,json,math,os,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from robometer_training.data import ProgressDataset,ProgressCollator
from robometer_training.model import load_processor,load_model,load_adapter,current_logits,progress_loss,progress_values
class Collator:
 def __init__(self,processor):self.encode=ProgressCollator(processor).encode
 def __call__(self,rows):return self.encode(rows),rows
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--adapter',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);torch.set_num_threads(1)
 c=json.loads(Path(args.config).read_text());processor=load_processor(c['model_path'],min_pixels=c['min_pixels'],max_pixels=c['max_pixels'])
 model,_=load_model(c['model_path'],processor);load_adapter(model,args.adapter);model.cuda().eval()
 rows=ProgressDataset(c['validation_data']).rows[rank::world]
 loader=DataLoader(rows,batch_size=c['training']['per_device_eval_batch_size'],shuffle=False,num_workers=4,persistent_workers=True,prefetch_factor=2,pin_memory=True,collate_fn=Collator(processor),multiprocessing_context='spawn')
 output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
 started=time.monotonic();count=0;loss_sum=absolute=squared=bias=0.0;torch.cuda.reset_peak_memory_stats()
 with (output/f'predictions-rank{rank}.jsonl').open('w') as stream:
  for batch,records in loader:
   batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=current_logits(model,batch)
   if not torch.isfinite(logits).all():raise RuntimeError('Non-finite validation logits')
   target=torch.tensor([r['pt'] for r in records],device='cuda',dtype=torch.float32)
   loss_sum+=float(progress_loss(logits,target))*len(records)
   values=progress_values(logits).tolist()
   for row,value in zip(records,values,strict=True):
    error=100*(value-row['pt']);absolute+=abs(error);squared+=error*error;bias+=error
    record={k:row[k] for k in ('id','episode_id','frame_index','task','history_frame_indices')}
    record.update(sample_index=rank+count*world,target_pt=row['pt'],predicted_pt=value)
    stream.write(json.dumps(record,ensure_ascii=False)+'\n');count+=1
   if count%200<8:
    stream.flush();(output/f'status-rank{rank}.json').write_text(json.dumps(dict(phase='validating',samples=count,total=len(rows))))
 report=dict(rank=rank,samples=count,loss_sum=loss_sum,absolute_error_sum_pp=absolute,squared_error_sum_pp2=squared,bias_sum_pp=bias,elapsed_seconds=time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
 (output/f'metrics-rank{rank}.json').write_text(json.dumps(report,indent=2));(output/f'status-rank{rank}.json').write_text(json.dumps(dict(phase='completed',samples=count,total=len(rows))))
 print('VALIDATION_RANK_DONE',json.dumps(report),flush=True)
if __name__=='__main__':main()
