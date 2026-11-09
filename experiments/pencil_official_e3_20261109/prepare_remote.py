from pathlib import Path
import json, shutil, hashlib

ROOT=Path('/cache/cvpzhr/setup/forest-robometer-pencil-official-e3-20261109')
OLD_F=Path('/cache/cvpzhr/setup/forest-pencil-spray-frome1-robomatched-20261109/pencil')
OLD_R=Path('/cache/cvpzhr/setup/robometer-pencil-case-clean-e3-20261109')
FOUT=Path('/cache/cvpzhr/outputs/forest-pencil-frombase-matched-e3-20261109')
ROUT=Path('/cache/cvpzhr/outputs/robometer-pencil-fromofficial-clean-2gpu-e3-20261109')
FEVAL=Path('/cache/cvpzhr/evaluations/forest-pencil-frombase-matched-e3-cleanval-20261109')
assert not ROOT.exists() and not FOUT.exists() and not ROUT.exists() and not FEVAL.exists()
ROOT.mkdir(parents=True)
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def replace(text,old,new):
    assert old in text,old[:100]
    return text.replace(old,new)
for name,old in [('forest',OLD_F),('robometer',OLD_R)]:
    s=ROOT/name;s.mkdir();shutil.copytree(old/'code',s/'code',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    save(s/'source-provenance.json',{'source_setup':str(old),'initialization_changed':True,'dataset_and_labels_unchanged':True})

f=json.loads((OLD_F/'training.config.json').read_text())
for k in ['initial_checkpoint','initial_checkpoint_sha256']:f.pop(k,None)
f.update(output_dir=str(FOUT),setup_dir=str(ROOT/'forest'),initialization_policy='fresh_base')
f['run_protocol'].update(initialization='fresh Qwen3.5-4B; new LoRA, history projection and progress head; no fine-tuned checkpoint',source_epoch=0,source_step=0,gpus=[0,1])
save(ROOT/'forest/training.config.json',f)
path=ROOT/'forest/code/mem_lite/train_stage3.py';text=path.read_text()
marker='    if checkpoint is None and config.get("initial_checkpoint"):\n'
audit='''    if checkpoint is None and config.get("initialization_policy") == "fresh_base":
        assert not config.get("initial_checkpoint")
        fresh = trainable_state(model)
        assert set(fresh) == {"m3_lora", "history_projection", "progress_head"}
        b_tensors = [v for k,v in fresh["m3_lora"].items() if "lora_B" in k]
        assert b_tensors and all(torch.count_nonzero(v).item() == 0 for v in b_tensors)
        fingerprints = {}
        for component, tensors in fresh.items():
            digest = hashlib.sha256()
            for name,tensor in sorted(tensors.items()):
                digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            fingerprints[component] = digest.hexdigest()
        audit = {"initialization": "fresh Qwen3.5-4B base", "model_path": config["model_path"],
                 "initial_checkpoint": None, "fresh_lora_B_all_zero": True,
                 "fresh_components": list(fresh), "initial_trainable_sha256": fingerprints,
                 "optimizer_reset": True, "scheduler_reset": True, "training_counter_reset": True, "rank": rank}
        audit_dir = Path(config.get("initialization_audit_dir", config["setup_dir"]))
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / f"initialization-rank{rank}.json").write_text(json.dumps(audit, indent=2)+"\\n")
        del fresh
'''
path.write_text(replace(text,marker,audit+marker))
for name in ['evaluate_final.py','plot_validation.py','verify_checkpoints.py']:
    text=(OLD_F/name).read_text()
    text=text.replace('/cache/cvpzhr/outputs/forest-pencil-frome1-robomatched-e3-20261109',str(FOUT))
    text=text.replace('/cache/cvpzhr/evaluations/forest-pencil-frome1-e3-cleanval-20261109',str(FEVAL))
    text=text.replace('从 Count/Order/State epoch 1 继续微调三轮','从 Qwen3.5-4B 基座重新初始化并训练三轮')
    if name=='verify_checkpoints.py':
        text=replace(text," assert x['config']['initial_checkpoint']==str(Path('/cache/cvpzhr/evaluations/forest-robometer-e1-augval-pearson-20261105/forest-epoch1.pt'))", " assert not x['config'].get('initial_checkpoint') and x['config']['initialization_policy']=='fresh_base'")
    (ROOT/'forest'/name).write_text(text)

r=json.loads((OLD_R/'training.config.json').read_text())
for k in ['initial_checkpoint','benchmark_reference','runtime_memory_adjustment']:r.pop(k,None)
r.update(output_dir=str(ROUT),setup_dir=str(ROOT/'robometer'),save_steps=600,save_total_limit=3,
         training_initialization='official Robometer-4B weights including official progress head; fresh language LoRA; new optimizer/scheduler/counters',
         initialization_policy='official_weights',initial_model_revision='beef63bc914c5c189329d49c6d712d96d632aa34',steps_per_epoch=600)
save(ROOT/'robometer/training.config.json',r)
path=ROOT/'robometer/code/robometer_training/train.py'
path.write_text(replace(path.read_text(),"save_strategy='steps', save_steps=config['eval_steps'],", "save_strategy='steps', save_steps=config.get('save_steps',config['eval_steps']),"))
text=(OLD_R/'train_from_checkpoint.py').read_text()
text=text.replace('"""Warm-start weights, train three new epochs, evaluate once at the final step."""','"""Fine-tune official Robometer weights with fresh LoRA, then evaluate epoch three."""')
text=text.replace('from robometer_training.train import make_trainer, compute_metrics','from robometer_training.train import make_trainer, compute_metrics, RunMonitor')
text=replace(text,"    args=parser.parse_args();config=json.loads(args.config.read_text())", "    parser.add_argument('--smoke',action='store_true')\n    args=parser.parse_args();config=json.loads(args.config.read_text())\n    assert not config.get('initial_checkpoint') and config['initialization_policy']=='official_weights'\n    if args.smoke:\n        config['output_dir']=str(Path(config['setup_dir'])/'smoke/output')\n        config['training'].update(max_steps=4,logging_steps=1)\n        config['eval_strategy']='no'")
start=text.index("    checkpoint=Path(config['initial_checkpoint']);output=Path(config['output_dir'])")
end=text.index('    trainer=make_trainer(config,model,processor)',start)
initial='''    output=Path(config['output_dir'])
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
'''
text=text[:start]+initial+text[end:]
marker='    rows=trainer.eval_dataset.rows\n'
smoke='''    if args.smoke:
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
'''
text=replace(text,marker,smoke+marker)
text=text.replace("'initial_checkpoint':str(checkpoint),'final_checkpoint':str(final)","'initial_model':config['model_path'],'initialization':'official weights plus fresh LoRA','final_checkpoint':str(final)")
text=text.replace("metrics['samples']=len(rows);metrics['episodes']=len(episodes)","metrics['samples']=len(rows);metrics['episodes']=len(episodes)\n        metrics['pearson_defined_episodes']=len(correlations)\n        metrics['pearson_undefined_episodes']=len(episodes)-len(correlations)")
(ROOT/'robometer/train_official.py').write_text(text)
for name in ['postprocess_validation.py','plot_validation_curves.py']:
    text=(OLD_R/name).read_text()
    text=text.replace('从原 1 epoch 检查点继续微调 3 epoch','从官方 Robometer-4B 原始权重微调 3 epoch')
    text=text.replace("m['eval_pearson']", "m['eval_episode_mean_pearson']")
    text=text.replace('Pearson      ', '平均 Pearson ')
    (ROOT/'robometer'/name).write_text(text)

policy={'tasks':{
    'forest':{'gpus':[0,1],'setup_dir':str(ROOT/'forest'),'output_dir':str(FOUT),'evaluation_dir':str(FEVAL),'python':'/cache/cvpzhr/envs/forest/bin/python','initialization':'fresh Qwen3.5-4B'},
    'robometer':{'gpus':[2,3],'setup_dir':str(ROOT/'robometer'),'output_dir':str(ROUT),'evaluation_dir':str(ROUT/'evaluation'),'python':'/cache/cvpzhr/envs/robometer/bin/python','initialization':'official Robometer-4B'}},
    'epochs':3,'train_episodes':28,'train_samples':10789,'validation_episodes':7,'validation_samples':2177,
    'global_batch_size':18,'learning_rate':1e-4,'warmup_ratio':.03,'scheduler':'cosine','seed':42,'augmentation':'none',
    'configuration_choice':'User confirmed latest matched settings; only initialization changes',
    'save_each_epoch':True,'final_evaluation_epoch':3,'smoke_weights_reused':False,'preserve_previous_results':True,
    'robometer_upstream':'/cache/cvpzhr/code/robometer-current3-w8-e1-20261104/third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'}
save(ROOT/'launch-policy.json',policy)
print(json.dumps({'status':'prepared','setup':str(ROOT),'outputs':[str(FOUT),str(ROUT)]}))
