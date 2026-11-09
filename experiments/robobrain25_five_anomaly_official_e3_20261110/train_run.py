"""Native GRM answer SFT, fresh official initialization, three retained epoch checkpoints."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
import torch
from transformers import TrainerCallback,TrainingArguments,set_seed
from grm import Dataset,Collator,load_model,load_processor
from train import ForwardTrainer
from common import read,save,sha

def tensor_sha(model):
 h=hashlib.sha256()
 for name,p in sorted(model.named_parameters()):
  if p.requires_grad:
   h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

class Monitor(TrainerCallback):
 def __init__(self,folder,expected):self.folder=folder;self.expected=expected;self.start=time.time()
 def on_train_begin(self,args,state,control,**kw):
  assert args.world_size==2 and state.max_steps==self.expected,(args.world_size,state.max_steps,self.expected)
 def on_log(self,args,state,control,logs=None,**kw):
  if args.process_index:return
  rec=dict(step=state.global_step,total_steps=state.max_steps,epoch=state.epoch,timestamp=time.time());rec.update(logs or {})
  for key in ['loss','grad_norm']:
   if key in rec:assert math.isfinite(float(rec[key])),rec
  with (self.folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 def on_train_end(self,args,state,control,**kw):
  if not args.process_index:save(self.folder/'train-finished.json',dict(step=state.global_step,epoch=state.epoch,seconds=time.time()-self.start))

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path)
 a=p.parse_args();c=read(a.config);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
 assert int(os.environ['WORLD_SIZE'])==2 and c['initial_checkpoint'] is None
 out=Path(c['output_dir']);audit=Path(c['audit_dir']);audit.mkdir(parents=True,exist_ok=True)
 if a.resume:
  assert not a.smoke and read(a.resume/'complete.json')['complete']
  old=read(a.resume/'run_config.json');assert old==c,'Only exact config resumes allowed'
 else:assert not out.exists() or not any(out.iterdir()),out
 set_seed(c['seed']);processor=load_processor(c['model_path'],c['min_pixels'],c['max_pixels']);model=load_model(c['model_path'],lora=c['lora'])
 train=Dataset(c['data_dir'],'train');assert len(train)==c['counts']['train']['samples']
 trainables=[(n,x) for n,x in model.named_parameters() if x.requires_grad]
 assert trainables and all('lora_' in n and '.language_model.' in n for n,x in trainables)
 zero=all(torch.count_nonzero(x)==0 for n,x in trainables if 'lora_B' in n);assert zero
 init=dict(initialization='official RoboBrain2.5-4B with fresh language LoRA',model_path=c['model_path'],official_revision=c['model_revision'],
  initial_checkpoint=None,fresh_lora_B_all_zero=True,trainable_sha256=tensor_sha(model),trainable_parameters=sum(x.numel() for _,x in trainables),
  only_language_lora_trainable=True,base_vision_lm_head_frozen=True,optimizer_reset=not bool(a.resume),scheduler_reset=not bool(a.resume),
  counters_reset=not bool(a.resume),resume_checkpoint=str(a.resume) if a.resume else None,rank=rank)
 save(audit/f'{"resume-" if a.resume else ""}initialization-rank{rank}.json',init)
 options=dict(c['training']);gc=options.pop('gradient_checkpointing');options.pop('per_device_eval_batch_size',None)
 arguments=TrainingArguments(output_dir=str(out),bf16=True,gradient_checkpointing=gc,gradient_checkpointing_kwargs={'use_reentrant':False},
  eval_strategy='no',save_strategy='no' if a.smoke else 'steps',save_steps=c['steps_per_epoch'],save_total_limit=None,
  max_steps=4 if a.smoke else -1,remove_unused_columns=False,label_names=['target_pt'],report_to='none',seed=c['seed'],
  ddp_find_unused_parameters=False,logging_nan_inf_filter=False,dataloader_pin_memory=True,**options)
 trainer=ForwardTrainer(model=model,args=arguments,config=c,processing_class=processor,train_dataset=train,
  data_collator=Collator(processor,c['max_length']),callbacks=[Monitor(out,4 if a.smoke else c['total_steps'])])
 out.mkdir(parents=True,exist_ok=True)
 if rank==0:save(out/'run_config.json',c)
 torch.cuda.reset_peak_memory_stats();start=time.time();result=trainer.train(resume_from_checkpoint=str(a.resume) if a.resume else None)
 peak=torch.cuda.max_memory_reserved()/2**30
 assert math.isfinite(result.training_loss) and trainer.state.global_step==(4 if a.smoke else c['total_steps'])
 if a.smoke:
  save(audit/f'smoke-rank{rank}.json',dict(status='passed',steps=4,loss=result.training_loss,seconds=time.time()-start,peak_reserved_gib=peak,weights_reused=False))
 else:
  trainer.save_state()
  if rank==0:
   assert (out/f"checkpoint-{c['total_steps']}"/'complete.json').exists()
   save(out/'training-complete.json',dict(status='completed',epochs=3,step=trainer.state.global_step,seconds=time.time()-start,peak_reserved_gib=peak,
     final_checkpoint=str(out/f"checkpoint-{c['total_steps']}"),initialization='official original weights; fresh LoRA',validation_started=False))
 trainer.accelerator.wait_for_everyone()
 if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
if __name__=='__main__':main()
