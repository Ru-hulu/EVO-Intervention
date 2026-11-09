from pathlib import Path
import json,sys,torch,hashlib
S=Path(__file__).resolve().parent
OUT=Path('/cache/cvpzhr/outputs/forest-spray-frombase-matched-e3-20261109')
sys.path.insert(0,str(S/'code'))
from mem_lite.train_stage3 import CHECKPOINT_FORMAT
rows=[];signature=None;last_step=0
for epoch in range(1,4):
 p=OUT/f'epoch-{epoch:02d}.pt';x=torch.load(p,map_location='cpu',weights_only=True)
 assert x['format']==CHECKPOINT_FORMAT and x['epoch']==epoch and x['next_batch']==0 and x['step']>last_step
 assert x['config']['epochs']==3 and not x['config']['keep_only_latest']
 assert len(x['rng_by_rank'])==2 and x['config']['learning_rate']==1e-4
 assert x['step']==epoch*x['config']['expected_steps_per_epoch']
 assert x['scheduler']['last_epoch']==x['step']
 assert not x['config'].get('initial_checkpoint') and x['config']['initialization_policy']=='fresh_base'
 assert x['config']['warmup_steps']==38
 assert not x['config'].get('validation_sample_files')
 if signature is None:signature=x['config']['data_signature']
 assert x['config']['data_signature']==signature
 assert set(x['model'])=={'m3_lora','history_projection','progress_head'}
 assert all(torch.isfinite(v).all() for state in x['model'].values() for v in state.values())
 last_step=x['step'];del x
 digest=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
 rows.append({'epoch':epoch,'checkpoint':str(p),'step':last_step,'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
(S/'checkpoint-verification.json').write_text(json.dumps({'status':'passed','archives':rows},indent=2)+'\n')
print(json.dumps(rows),flush=True)
