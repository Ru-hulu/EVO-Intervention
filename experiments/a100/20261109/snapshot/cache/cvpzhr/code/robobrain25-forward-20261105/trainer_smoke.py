"""Two-rank integration: complete save before generation validation, then resume."""
from functools import partial
import gc
import json
import os
from pathlib import Path
import torch
from transformers import TrainingArguments, set_seed
from grm import load_model,load_processor,Collator
from train import ForwardTrainer,compute_metrics,FinishEpoch
from periodic import ValidationProgress

ROOT=Path('/cache/cvpzhr/setup/robobrain25-forward-20261105')


def main():
    torch.set_num_threads(2);set_seed(42)
    config=json.loads((ROOT/'config.draft.json').read_text())
    assert json.loads((ROOT/'gpu-smoke.json').read_text())['status']=='passed'
    rows=json.loads((ROOT/'smoke-rows.json').read_text())
    out=ROOT/'trainer-smoke'
    config=dict(config,output_dir=str(out),purpose='implementation-smoke-only')
    processor=load_processor(config['model_path'],config['min_pixels'],config['max_pixels'])
    model=load_model(config['model_path'],lora=config['lora'])
    args=TrainingArguments(output_dir=str(out),max_steps=2,per_device_train_batch_size=1,
        per_device_eval_batch_size=1,learning_rate=1e-4,bf16=True,gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},eval_strategy='steps',eval_steps=1,
        save_strategy='steps',save_steps=1,save_total_limit=2,metric_for_best_model='mae_pp',
        greater_is_better=False,load_best_model_at_end=False,remove_unused_columns=False,
        label_names=['target_pt'],report_to='none',ddp_find_unused_parameters=False,
        dataloader_num_workers=0,disable_tqdm=True,logging_steps=1,seed=42,optim='adamw_torch_fused')
    trainer=ForwardTrainer(model=model,args=args,config=config,processing_class=processor,
        train_dataset=rows,eval_dataset=rows[:3],data_collator=Collator(processor,config['max_length']),
        compute_metrics=partial(compute_metrics,metadata=rows[:3]),callbacks=[FinishEpoch(),ValidationProgress()])
    trainer.train()
    assert trainer.state.global_step==2
    trainer.accelerator.wait_for_everyone()
    trainer.args.max_steps=3
    trainer.train(resume_from_checkpoint=str(out/'checkpoint-2'))
    assert trainer.state.global_step==3
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        events=[json.loads(s) for s in (out/'lifecycle.jsonl').read_text().splitlines()]
        validations=[(i,e) for i,e in enumerate(events) if e['event']=='validation_started']
        for i,event in validations:
            assert any(e['event']=='checkpoint_saved' and e['step']==event['step'] for e in events[:i])
        receipt=json.loads((out/'checkpoint-3/complete.json').read_text())
        assert receipt['saved_before_validation']
        assert (out/'checkpoint-3/validation.json').exists()
        assert (out/'last').resolve()==out/'checkpoint-3'
        report=dict(status='passed',world_size=int(os.environ['WORLD_SIZE']),
            initial_steps=2,resumed_step=3,generation_validation_samples=3,
            checkpoint_saved_before_every_validation=True,optimizer_scheduler_rng_restored=True,
            validation_runs=len(validations),formal_training_started=False)
        (ROOT/'trainer-smoke.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    trainer.accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
