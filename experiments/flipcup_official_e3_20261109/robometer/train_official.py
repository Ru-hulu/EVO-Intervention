"""Fine-tune official Robometer weights with fresh LoRA, then evaluate epoch three."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from peft import get_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights
from safetensors.torch import load_file
import torch
from transformers import set_seed

from robometer_training.model import attach_lora, load_adapter, load_model, load_processor, progress_values
from robometer_training.train import make_trainer, compute_metrics, RunMonitor


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def pearson(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    assert not config.get('initial_checkpoint') and config['initialization_policy']=='official_weights'
    if args.smoke:
        config['output_dir']=str(Path(config['setup_dir'])/'smoke/output')
        config['training'].update(max_steps=4,logging_steps=1)
        config['eval_strategy']='no'
    rank=int(os.environ.get('RANK','0'))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.multiprocessing.set_start_method('spawn',force=True)
    set_seed(config['seed'])
    output=Path(config['output_dir'])
    assert not output.exists() or not any(output.iterdir()), 'Nonempty output: explicit recovery audit required'
    processor=load_processor(config['model_path'],min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
    model,_=load_model(config['model_path'],processor)  # Strict load of every published tensor.
    official_head={k:v.detach().cpu().float().clone() for k,v in model.progress_head.state_dict().items()}
    attach_lora(model,**config['lora'])
    assert all(torch.equal(model.progress_head.state_dict()[k].cpu(),v) for k,v in official_head.items())
    b_tensors=[p for name,p in model.named_parameters() if 'lora_B' in name]
    assert b_tensors and all(torch.count_nonzero(p).item()==0 for p in b_tensors)
    assert all(('lora_' in name or name.startswith('progress_head.')) for name,p in model.named_parameters() if p.requires_grad)
    digest=hashlib.sha256()
    for name,tensor in sorted(official_head.items()):
        digest.update(name.encode());digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    audit_dir=Path(config['setup_dir'])/('smoke' if args.smoke else '')
    write_json(audit_dir/f'initialization-rank{rank}.json',{
        'initial_model':config['model_path'],'official_revision':config['initial_model_revision'],
        'initial_checkpoint':None,'strict_original_weight_load':True,'official_progress_head_exactly_preserved':True,
        'official_progress_head_sha256':digest.hexdigest(),'fresh_lora_B_all_zero':True,
        'optimizer_reset':True,'scheduler_reset':True,'training_counter_reset':True,'rank':rank})
    del official_head
    trainer=make_trainer(config,model,processor)
    assert len(trainer.train_dataset)==config['expected_train_samples'] and len(trainer.eval_dataset)==config['expected_validation_samples']
    for dataset in [trainer.train_dataset,trainer.eval_dataset]:
        assert all(r['task_id']==config['task'] and r['kind']=='original' and r['pt_policy']=='none' for r in dataset.rows)
    assert trainer.state.global_step==0
    if args.smoke:
        trainer.remove_callback(RunMonitor)
        trainer.args.save_strategy='no'
        trainer.args.eval_strategy='no'
        torch.cuda.reset_peak_memory_stats()
        started=time.time();result=trainer.train()
        assert trainer.state.global_step==4
        assert all(np.isfinite(v) for row in trainer.state.log_history for k,v in row.items() if k in ['loss','grad_norm'])
        assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
        write_json(audit_dir/f'smoke-verification-rank{rank}.json',{
            'status':'passed','steps':4,'weights_reused':False,'seconds':time.time()-started,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'train_loss':result.training_loss})
        trainer.accelerator.wait_for_everyone()
        return
    rows=trainer.eval_dataset.rows

    def evaluate_and_save(prediction):
        predicted=progress_values(torch.from_numpy(prediction.predictions)).numpy()
        target=prediction.label_ids
        assert len(predicted)==len(target)==len(rows)==config['expected_validation_samples']
        assert np.isfinite(predicted).all() and np.isfinite(target).all()
        metrics=compute_metrics(prediction)
        metrics['bias_pp']=float(100*(predicted-target).mean())
        corr=pearson(predicted,target)
        if corr is not None:metrics['pearson']=corr
        grouped=defaultdict(list)
        for i,row in enumerate(rows):
            assert abs(float(target[i])-row['pt'])<1e-6
            grouped[row['episode_id']].append(i)
        episodes=[]
        for episode,indices in grouped.items():
            a,b=predicted[indices],target[indices];err=100*(a-b)
            episodes.append({'episode_id':episode,'samples':len(indices),'mae_pp':float(np.abs(err).mean()),
                             'rmse_pp':float(np.sqrt(np.mean(err**2))),'pearson':pearson(a,b)})
        correlations=[r['pearson'] for r in episodes if r['pearson'] is not None]
        if correlations:metrics['episode_mean_pearson']=float(np.mean(correlations))
        metrics['samples']=len(rows);metrics['episodes']=len(episodes)
        metrics['pearson_defined_episodes']=len(correlations)
        metrics['pearson_undefined_episodes']=len(episodes)-len(correlations)
        if trainer.is_world_process_zero():
            folder=output/'evaluation';folder.mkdir(exist_ok=True)
            partial=folder/'predictions.jsonl.tmp'
            with partial.open('w') as stream:
                for i,row in enumerate(rows):
                    record={'index':i,'id':row['id'],'episode_id':row['episode_id'],'task_id':row['task_id'],
                            'frame_index':row['frame_index'],'source_frame_index':row['source_frame_index'],
                            'target_pt':float(target[i]),'predicted_pt':float(predicted[i])}
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
            partial.replace(folder/'predictions.jsonl')
            write_json(folder/'episode_metrics.json',episodes)
        return metrics

    trainer.compute_metrics=evaluate_and_save
    if trainer.is_world_process_zero():
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/'run_config.json',config)
    trainer.accelerator.wait_for_everyone()
    started=time.time()
    result=trainer.train()  # Weights only; never restore the old optimizer or epoch counter.
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    assert trainer.state.global_step==config['expected_steps'] and abs(trainer.state.epoch-config['training']['num_train_epochs'])<1e-6
    if trainer.is_world_process_zero():
        last=json.loads((output/'last.json').read_text())
        assert last['complete'] and last['step']==config['expected_steps']
        final=Path(last['checkpoint'])
        evaluation=json.loads((final/'validation.json').read_text())
        assert evaluation['eval_samples']==config['expected_validation_samples'] and evaluation['eval_episodes']==config['expected_validation_episodes']
        write_json(output/'evaluation/metrics.json',evaluation)
        write_json(output/'report.json',{'status':'completed','task':config['task'],
                   'initial_model':config['model_path'],'initialization':'official weights plus fresh LoRA','final_checkpoint':str(final),
                   'additional_epochs':config['training']['num_train_epochs'],'steps':trainer.state.global_step,'train_samples':config['expected_train_samples'],
                   'validation_samples':config['expected_validation_samples'],'augmentation':'none','evaluation':evaluation,
                   'train_metrics':result.metrics,'elapsed_seconds':time.time()-started})
    trainer.accelerator.wait_for_everyone()


if __name__=='__main__':main()
