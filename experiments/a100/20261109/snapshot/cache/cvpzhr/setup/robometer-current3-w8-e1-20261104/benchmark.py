"""Disposable four-GPU optimizer benchmark and checkpoint/resume smoke."""
import argparse,json,os,time
from pathlib import Path
import torch
from transformers import TrainerCallback,set_seed
from robometer_training.model import load_model,load_processor,attach_lora,progress_loss
from robometer_training.data import ProgressCollator
from robometer_training.train import make_trainer
from robometer_training.periodic import atomic_json

class Timing(TrainerCallback):
    def __init__(self,output,stop=None):
        self.output=Path(output);self.stop=stop;self.previous=None;self.records=[];self.eval_records={}
    def on_step_end(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        now=time.perf_counter()
        self.records.append(dict(step=state.global_step,
            seconds=None if self.previous is None else now-self.previous))
        self.previous=now
        if self.stop and state.global_step>=self.stop:
            control.should_training_stop=True
        return control
    def on_prediction_step(self,args,state,control,**kwargs):
        torch.cuda.synchronize()
        now=time.perf_counter()
        values=self.eval_records.setdefault(str(state.global_step),[])
        values.append(dict(time=now,seconds=None if not values else now-values[-1]['time']))
    def on_train_end(self,args,state,control,**kwargs):
        atomic_json(self.output/f'timing-rank{args.process_index}.json',
                    dict(rank=args.process_index,step=state.global_step,records=self.records,evaluation=self.eval_records,
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                         peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--resume')
    p.add_argument('--output')
    p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.multiprocessing.set_start_method('spawn',force=True)
    c=json.loads(Path(a.config).read_text())
    if a.output:c['output_dir']=a.output
    out=Path(c['output_dir'])
    out.mkdir(parents=True,exist_ok=True)
    set_seed(c['seed'])
    processor=load_processor(c['model_path'],min_pixels=c['min_pixels'],max_pixels=c['max_pixels'])
    model,_=load_model(c['model_path'],processor)
    attach_lora(model,**c['lora'])
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert all('lora_' in n or n.startswith('progress_head.') for n,p in trainable)
    assert not any(p.requires_grad for p in model.model.visual.parameters())
    stress_report=None
    if not a.resume:
        row=json.loads((Path(a.config).parent/'stress-row.json').read_text())
        model.cuda().train()
        model.model.visual.eval()
        batch=ProgressCollator(processor)([row]*c['training']['per_device_train_batch_size'])
        batch={k:v.cuda() for k,v in batch.items()}
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output,_=model(**{k:v for k,v in batch.items() if k!='labels'},sample_type='progress')
            assert output.progress_logits['A'].shape[1]==8
            loss=progress_loss(output.progress_logits['A'][:,-1].float(),batch['labels'])
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in trainable)
        assert all(p.grad is None for p in model.model.visual.parameters())
        torch.cuda.synchronize()
        stress_report=dict(images=10,progress_tokens=8,all_trainable_gradients_finite=True,
            visual_frozen=True,loss=float(loss.detach()),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        model.zero_grad(set_to_none=True)
        del batch,output,loss
        torch.cuda.empty_cache()
    trainer=make_trainer(c,model,processor)
    timer=Timing(out,a.stop_after)
    trainer.add_callback(timer)
    torch.cuda.reset_peak_memory_stats()
    started=time.time()
    result=trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_state()
    if trainer.is_world_process_zero():
        atomic_json(out/'smoke-result.json',
            dict(status='passed',step=trainer.state.global_step,started_at=started,
                 seconds=time.time()-started,resumed_from=a.resume,
                 best_checkpoint=trainer.state.best_model_checkpoint,
                 trainable_parameters=sum(p.numel() for n,p in trainable),
                 trainable_scope='language LoRA and official progress head',stress=stress_report,
                 metrics=result.metrics,scheduler_last_epoch=trainer.lr_scheduler.state_dict().get('last_epoch'),optimizer_state_entries=len(trainer.optimizer.state),formal_training_started=False))
        print('SMOKE_DONE',json.dumps(dict(step=trainer.state.global_step,seconds=time.time()-started)),flush=True)
if __name__=='__main__':main()
