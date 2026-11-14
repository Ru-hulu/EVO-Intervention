"""Two-rank LoRA training with native cached inputs and audited final checkpoints."""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
from functools import lru_cache

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from .common import *
from .data import CachedCollator, Dataset


@lru_cache(maxsize=1)
def robometer_api():
    sys.path.insert(0,'/cache/cvpzhr/setup/three-model-e1-cleanval-20261108/code/robometer-upstream')
    path='/cache/cvpzhr/setup/nine-model-zeroshot-cleanval-voc-20261113/runtime/robometer_training/model.py'
    spec=importlib.util.spec_from_file_location('recover_native_robometer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def load_model(config):
    if config['name']=='robometer-4b':
        api=robometer_api()
        processor=api.load_processor(config['model_path'],min_pixels=4096,max_pixels=65536)
        model,_=api.load_model(config['model_path'],processor)
        # The native trainer promotes the head from BF16 to FP32; compare exact numeric
        # values in FP32 on both sides instead of rejecting this lossless dtype cast.
        head=tensor_digest((n,v.float()) for n,v in model.progress_head.state_dict().items())
        api.attach_lora(model,rank=16,alpha=32,dropout=.05)
        assert head==tensor_digest(model.progress_head.state_dict().items())
        original=dict(strict_official_weight_load=True,official_progress_head_sha256=head)
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig,get_peft_model
        native=read(CACHE_SETUP/f"{config['name']}-native-provenance.json")['config']
        processor=AutoProcessor.from_pretrained(config['model_path'],local_files_only=True,
            **{k:native[k] for k in ('min_pixels','max_pixels') if k in native})
        model,info=AutoModelForImageTextToText.from_pretrained(config['model_path'],local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        assert not info.get('missing_keys') and not info.get('unexpected_keys') and not info.get('mismatched_keys'),info
        model.requires_grad_(False)
        targets=[n for n,p in model.named_modules() if '.language_model.' in n and
                 n.rsplit('.',1)[-1] in ('q_proj','k_proj','v_proj','o_proj') and isinstance(p,torch.nn.Linear)]
        assert targets
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',
            target_modules=targets,task_type='CAUSAL_LM'))
        original=dict(strict_official_weight_load=True,language_attention_lora_targets=len(targets),
                      auxiliary_progress_head_used=False)
    model.config.use_cache=False
    if hasattr(model.config,'text_config'):model.config.text_config.use_cache=False
    b=[p for n,p in model.named_parameters() if 'lora_B' in n]
    assert b and all(torch.count_nonzero(p).item()==0 for p in b)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert all('lora_' in n or (config['name']=='robometer-4b' and n.startswith('progress_head.')) for n,p in trainable)
    assert not any(p.requires_grad for n,p in model.named_parameters() if '.visual.' in n)
    original.update(fresh_lora_B_all_zero=True,trainable_parameters=sum(p.numel() for n,p in trainable),
        initial_trainable_sha256=tensor_digest(trainable),official_revision=config['official_revision'],
        rank=int(os.environ['RANK']),selection_sha256=config['selection_sha256'],
        initial_adapter=None,optimizer_reset=True)
    return model,processor,original


class Monitor(TrainerCallback):
    def __init__(self,config,folder): self.config,self.folder,self.started=config,folder,time.time()

    def on_log(self,args,state,control,logs=None,**kwargs):
        values=dict(logs or {})
        for key in ('loss','grad_norm'):
            if key in values: assert math.isfinite(values[key]), f'Nonfinite {key}'
        if state.is_world_process_zero:
            value=dict(values)
            value.update(status='running',step=state.global_step,total_steps=state.max_steps,epoch=state.epoch,
                samples=2550,updated=time.time(),elapsed=time.time()-self.started)
            save(self.folder/'progress.json',value)
            with (self.folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step%10==0:
            free,total=torch.cuda.mem_get_info()
            assert free>16*1024**3,'Stop this job before exhausting GPU headroom'
        stop=torch.tensor(int((ROOT/'resource-stop.json').exists()),device='cuda')
        if torch.distributed.is_initialized():torch.distributed.all_reduce(stop,op=torch.distributed.ReduceOp.MAX)
        if stop.item():control.should_training_stop=True;control.should_save=True
        return control


class ProgressTrainer(Trainer):
    def __init__(self,*args,config,**kwargs):
        self.run_config=config
        super().__init__(*args,**kwargs)
        self.model_accepts_loss_kwargs=False

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        if self.run_config['name']=='robometer-4b':
            labels=inputs.pop('labels');api=robometer_api()
            logits=api.current_logits(model,inputs);loss=api.progress_loss(logits,labels)
        else: loss=model(**inputs).loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        return (loss,{'loss':loss}) if return_outputs else loss

    def _save(self,output_dir=None,state_dict=None):
        output=Path(output_dir or self.args.output_dir);output.mkdir(parents=True,exist_ok=True)
        if self.run_config['name']=='robometer-4b':robometer_api().save_adapter(self.model,output)
        else:self.model.save_pretrained(output,save_embedding_layers=False)
        self.processing_class.save_pretrained(output/'processor')
        state={n:p.detach().cpu().contiguous() for n,p in self.model.named_parameters() if p.requires_grad}
        assert all(torch.isfinite(p).all() for p in state.values())
        torch.save(state,output/'trainable-state.pt')
        save(output/'trainable-state.json',dict(sha256=tensor_digest(state.items()),parameters=list(state)))
        save(output/'run_config.json',self.run_config)
        torch.save(self.args,output/'training_args.bin')

    def _load_from_checkpoint(self,resume_from_checkpoint,model=None):
        checkpoint=Path(resume_from_checkpoint)
        assert read(checkpoint/'run_config.json')==self.run_config,'Resume config differs'
        state=torch.load(checkpoint/'trainable-state.pt',map_location='cpu',weights_only=True)
        model=model or self.model
        current={n:p for n,p in model.named_parameters() if p.requires_grad}
        assert set(current)==set(state)
        with torch.no_grad():
            for name,p in current.items():p.copy_(state[name])
        assert tensor_digest(current.items())==read(checkpoint/'trainable-state.json')['sha256']


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--pilot',action='store_true')
    p.add_argument('--resume');args=p.parse_args();config=read(args.config)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE'])==2
    torch.cuda.set_device(local);torch.cuda.set_per_process_memory_fraction(config['allocator_fraction'])
    torch.set_num_threads(4);set_seed(config['seed'])
    assert sha(config['train_data'])==config['train_sha256']
    for path,digest in config['weights_sha256'].items():assert sha(path)==digest
    folder=Path(config['setup_dir'])/('pilot' if args.pilot else 'formal');folder.mkdir(parents=True,exist_ok=True)
    output=folder/'checkpoints' if args.pilot else Path(config['output_dir'])
    assert args.resume or not output.exists() or not any(output.iterdir()),'Refuse nonempty output'
    model,processor,audit=load_model(config)
    save(folder/f'initialization-rank{rank}.json',audit)
    dataset=Dataset(config['train_data']);assert len(dataset)==2550
    if args.pilot:
        from safetensors.torch import load_file
        longest=max(dataset.rows,key=lambda r:load_file(r['prompt_tensor_file'])['input_ids'].numel())
        shortest=min(dataset.rows,key=lambda r:load_file(r['prompt_tensor_file'])['input_ids'].numel())
        # Include real padding masks: identical longest inputs alone can select a
        # cheaper attention kernel and understate the memory of shuffled batches.
        dataset.rows=[longest,shortest]*(config['microbatch']*4)
    expected=math.ceil(math.ceil(2550/2/config['microbatch'])/config['accumulation'])
    ta=TrainingArguments(output_dir=str(output),num_train_epochs=config['epochs'],
        max_steps=3 if args.pilot else -1,per_device_train_batch_size=config['microbatch'],
        gradient_accumulation_steps=1 if args.pilot else config['accumulation'],learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup'],lr_scheduler_type=config['schedule'],weight_decay=.01,max_grad_norm=1,
        bf16=True,gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={'use_reentrant':False},optim='adamw_torch_fused',
        eval_strategy='no',save_strategy='steps',save_steps=3 if args.pilot else 100,save_total_limit=2,
        logging_steps=1 if args.pilot else 5,logging_first_step=True,logging_nan_inf_filter=False,
        report_to='none',disable_tqdm=True,seed=42,data_seed=42,remove_unused_columns=False,label_names=['labels'],
        dataloader_num_workers=config['num_workers'],dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,ddp_timeout=1800)
    trainer=ProgressTrainer(model=model,args=ta,config=config,train_dataset=dataset,
        data_collator=CachedCollator(processor.tokenizer,config['name'],config['max_length']),
        processing_class=processor,callbacks=[Monitor(config,folder)])
    torch.cuda.reset_peak_memory_stats();started=time.time()
    result=trainer.train(resume_from_checkpoint=args.resume)
    trainer._save_checkpoint(trainer.model,trial=None)
    trainer.save_state();trainer.accelerator.wait_for_everyone()
    target_steps=3 if args.pilot else expected
    assert trainer.state.global_step==target_steps,(trainer.state.global_step,target_steps)
    if not args.pilot:assert abs(trainer.state.epoch-1)<1e-6
    state={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert all(torch.isfinite(p).all() for p in state.values())
    final_digest=tensor_digest(state.items());assert final_digest!=audit['initial_trainable_sha256']
    report=dict(status='passed' if args.pilot else 'completed',model=config['name'],rank=rank,
        steps=target_steps,epoch=trainer.state.epoch,seconds=time.time()-started,metrics=result.metrics,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
        final_trainable_sha256=final_digest,checkpoint=str(output/f'checkpoint-{target_steps}'),
        formal_weights_from_pilot=False,pilot=args.pilot,config_sha256=sha(args.config))
    save(folder/f'completion-rank{rank}.json',report)
    trainer.accelerator.wait_for_everyone()
    if rank==0:
        pair=[read(folder/f'completion-rank{i}.json') for i in (0,1)]
        assert pair[0]['final_trainable_sha256']==pair[1]['final_trainable_sha256']
        ckpt=Path(report['checkpoint'])
        for name in ('optimizer.pt','scheduler.pt','trainer_state.json','rng_state_0.pth','rng_state_1.pth'):
            assert (ckpt/name).stat().st_size>0,name
        assert torch.load(ckpt/'optimizer.pt',map_location='cpu',weights_only=True)['state']
        saved=torch.load(ckpt/'trainable-state.pt',map_location='cpu',weights_only=True)
        assert tensor_digest(saved.items())==final_digest
        saved_audit=[read(folder/f'initialization-rank{i}.json') for i in (0,1)]
        assert saved_audit[0]['initial_trainable_sha256']==saved_audit[1]['initial_trainable_sha256']
        save(folder/'verification.json',dict(status='passed',pilot=args.pilot,steps=target_steps,
            checkpoint=str(ckpt),trainable_state_sha256=sha(ckpt/'trainable-state.pt'),
            optimizer_scheduler_two_rank_rng=True,replicas_equal=True,fresh_official_initialization=True,
            train_input_sha256=config['train_sha256'],selection_sha256=config['selection_sha256'],
            expected_samples=2550,expected_episodes=10,epoch=trainer.state.epoch,
            peak_reserved_gib=max(r['peak_reserved_gib'] for r in pair),
            samples_per_second=result.metrics.get('train_samples_per_second'),completed=time.time()))
    trainer.accelerator.wait_for_everyone()


if __name__=='__main__':main()
