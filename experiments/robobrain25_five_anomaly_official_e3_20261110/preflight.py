from pathlib import Path
import ast,json,time
from common import *
def main():
 assert read(ROOT/'cache-status.json')['phase']=='completed'
 receipt=read('/cache/cvpzhr/setup/download-robobrain25-4b-20261105/complete.json');assert receipt['revision']=='ec4b3bf5e969b48ad17df432b479c10402a0079b'
 weights={}
 for item in receipt['files']:
  p=Path(receipt['target'])/item['name'];assert p.stat().st_size==item['size'] and sha(p)==item['sha256'];weights[item['name']]=item['sha256']
 counts={};sources={}
 for task in TASKS:
  c=config(task);d=Path(c['data_dir']);m=read(d/'manifest.json');assert m['status']=='completed' and m['counts']==c['counts']
  assert c['initial_checkpoint'] is None and c['reference_global_batch_size']==18 and c['training']['num_train_epochs']==3
  assert c['training']['learning_rate']==1e-4 and c['training']['warmup_ratio']==.03 and c['training']['lr_scheduler_type']=='cosine'
  assert not Path(c['output_dir']).exists()
  for n,h in m['input_files_sha256'].items():assert sha(d/n)==h
  assert m['uses_future_goal_reference'] and m['augmentation']=='none' and m['original_PT_preserved']
  assert m['max_input_tokens']<c['max_length']
  counts[task]=c['counts'];sources[task]=m['input_files_sha256']
 hashes={}
 for p in ROOT.glob('*.py'):ast.parse(p.read_text());hashes[p.name]=sha(p)
 hashes['prompt_template.txt']=sha(ROOT/'prompt_template.txt')
 for n in ['grm.py','periodic.py','metrics.py','prompt_template.txt']:
  assert sha(ROOT/n)==sha(Path('/cache/cvpzhr/code/robobrain25-forward-20261105')/n)
 save(ROOT/'preflight.json',dict(status='passed',official_weights_sha256=weights,official_revision=receipt['revision'],counts=counts,
   sources=sources,code_sha256=hashes,native_GRM_unchanged=True,existing_model_results_preserved=True,uses_future_goal_reference=True,created_at=time.time()))
 print('PREFLIGHT_PASSED',counts,flush=True)
if __name__=='__main__':main()
