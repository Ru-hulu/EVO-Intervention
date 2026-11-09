from pathlib import Path
import json,torch
from robometer_training.model import load_processor,load_model,attach_lora
from robometer_training.train import make_trainer
ROOT=Path('/cache/cvpzhr');RUN=ROOT/'setup/robometer-20261031'
c=json.loads((RUN/'recommended-config.json').read_text())
p=load_processor(c['model_path'],min_pixels=c['min_pixels'],max_pixels=c['max_pixels'])
m,_=load_model(c['model_path'],p);attach_lora(m,**c['lora'])
t=make_trainer(c,m,p,output_dir=str(RUN/'entry-check-output'),use_cpu=True)
t.create_optimizer()
expected={id(p) for p in m.parameters() if p.requires_grad};actual={id(p) for group in t.optimizer.param_groups for p in group['params']}
assert expected==actual
report=dict(status='passed',trainer_constructed=True,optimizer_parameter_groups_match_trainable_parameters=True,optimizer_steps=0,training_called=False,batch=t.args.per_device_train_batch_size,gradient_checkpointing=t.args.gradient_checkpointing,workers=t.args.dataloader_num_workers,persistent_workers=t.args.dataloader_persistent_workers,optimizer=str(t.args.optim))
(RUN/'training-entry-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
