from pathlib import Path
import copy,json,math,shutil
from common import *
assert ROOT.name=='robobrain25-five-anomaly-official-e3-20261110'
old=read('/cache/cvpzhr/setup/robobrain25-forward-20261105/config.json')
for task in TASKS:
 s=ROOT/task;assert not s.exists();s.mkdir()
 rc=read(f'/cache/cvpzhr/setup/forest-robometer-{task}-official-e3-20261109/robometer/training.config.json')
 rm=read(Path(rc['train_data']).parent/'manifest.json')
 c=copy.deepcopy(old);c.update(data_dir=f'/cache/cvpzhr/datasets/robobrain25-{task}-grm8-clean-20261110',
  output_dir=f'/cache/cvpzhr/outputs/robobrain25-{task}-fromofficial-clean-2gpu-e3-20261110',
  setup_dir=str(s),task=task,task_id=rc['task'],initial_checkpoint=None,initialization_policy='official_weights_fresh_lora',
  reference_world_size=2,reference_global_batch_size=18,augmentation='none',
  launch_authorization='User confirmed two concurrent tasks, two GPUs each, independent official starts, 3 epochs, global batch18, lr1e-4 warmup3%+cosine, no augmentation; save each epoch and validate/plot epoch3 only',
  counts={sp:dict(episodes=rm['splits'][sp]['episode_count'],samples=rm['splits'][sp]['samples']) for sp in ['train','validation']})
 c['steps_per_epoch']=math.ceil(c['counts']['train']['samples']/18);c['total_steps']=c['steps_per_epoch']*3
 c['training'].update(num_train_epochs=3,per_device_train_batch_size=9,per_device_eval_batch_size=9,gradient_accumulation_steps=1,
  dataloader_num_workers=4,dataloader_prefetch_factor=2,gradient_checkpointing=False)
 assert not Path(c['output_dir']).exists() and not Path(c['data_dir']).exists()
 save(s/'config.json',c);status(task,'pending_cache')
save(ROOT/'launch-policy.json',dict(tasks=TASKS,parallel_tasks=2,gpu_pairs=[[0,1],[2,3]],
 independent_official_initialization=True,official_revision=old['model_revision'],epochs=3,global_batch=18,lr=1e-4,warmup=.03,schedule='cosine',
 augmentation='none',validation='epoch3 only, saved before evaluation',all_epoch_checkpoints_retained=True,
 uses_future_goal_reference=True,smoke_candidates=[{'microbatch':9,'accumulation':1,'gradient_checkpointing':False},
 {'microbatch':3,'accumulation':3,'gradient_checkpointing':False},{'microbatch':3,'accumulation':3,'gradient_checkpointing':True}],
 smoke_weights_never_reused=True,other_tasks_not_interrupted=True))
print('CONFIGURED',TASKS)
