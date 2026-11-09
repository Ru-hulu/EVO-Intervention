from pathlib import Path
import json,math,hashlib
from robometer_training.data import ProgressDataset
S=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());cfg=read(S/'training.config.json');root=Path(cfg['train_data']).parent;manifest=read(root/'manifest.json');audit=read(root/'label-audit.json')
assert manifest['status']=='complete' and audit['status']=='passed' and manifest['augmentation']=='none'
assert cfg['training']['num_train_epochs']==3 and cfg['training']['learning_rate']==1e-4 and cfg['training']['warmup_ratio']==.03 and cfg['training']['lr_scheduler_type']=='cosine'
assert cfg['save_total_limit'] is None
ids={}
for split,n,e in [('train',7769,28),('validation',1744,7)]:
 rows=ProgressDataset(cfg['train_data'] if split=='train' else cfg['validation_data']).rows;assert len(rows)==n;ids[split]={r['episode_id'] for r in rows};assert len(ids[split])==e
 assert all(r['task_id']==cfg['task'] and r['kind']=='original' and r['pt_policy']=='none' and len(r['image'])==10 for r in rows)
 assert hashlib.sha256((root/f'{split}.jsonl').read_bytes()).hexdigest()==manifest['splits'][split]['sha256']
assert not ids['train'] & ids['validation']
steps=math.ceil(math.ceil(7769/2)/cfg['training']['per_device_train_batch_size']);assert steps==432 and steps*3==cfg['expected_steps']==cfg['eval_steps']
source=Path(cfg['initial_checkpoint']);assert abs(read(source/'trainer_state.json')['epoch']-1)<1e-6
for n in ['adapter_config.json','adapter_model.safetensors','progress_head.safetensors']:assert (source/n).is_file()
report={'status':'passed','train_samples':7769,'validation_samples':1744,'train_episodes':28,'validation_episodes':7,'world_size':2,'batch_per_gpu':9,'steps_per_epoch':steps,'total_steps':steps*3,'initial_checkpoint':str(source),'weights_only_warm_start':True,'augmentation':'none'}
(S/'training-preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
