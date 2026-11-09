from pathlib import Path
import json,hashlib,math,numpy as np
from collections import defaultdict
S=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());cfg=read(S/'training.config.json');out=Path(cfg['output_dir']);report=read(out/'report.json')
assert report['status']=='completed' and report['steps']==1296 and report['initial_checkpoint']==cfg['initial_checkpoint']
archives=[]
for epoch,step in [(1,432),(2,864),(3,1296)]:
 p=out/f'checkpoint-{step}';state=read(p/'trainer_state.json');complete=read(p/'complete.json')
 assert state['global_step']==step and abs(state['epoch']-epoch)<1e-6 and complete['complete'] and complete['saved_before_validation']
 hashes={}
 for name in complete['files']:
  f=p/name;assert f.is_file() and f.stat().st_size>0
  if name in ['adapter_model.safetensors','progress_head.safetensors']:hashes[name]=hashlib.sha256(f.read_bytes()).hexdigest()
 archives.append({'epoch':epoch,'step':step,'checkpoint':str(p),'sha256':hashes})
for rank in [0,1]:assert read(S/f'initialization-rank{rank}.json')['adapter_and_progress_head_exactly_restored']
expected=[json.loads(x) for x in Path(cfg['validation_data']).read_text().splitlines()];rows=[json.loads(x) for x in (out/'evaluation/predictions.jsonl').read_text().splitlines()];assert len(rows)==len(expected)==1744 and len({r['id'] for r in rows})==1744
groups=defaultdict(list)
for r,e in zip(rows,expected,strict=True):
 assert r['id']==e['id'] and r['source_frame_index']==e['source_frame_index'] and abs(r['target_pt']-e['pt'])<1e-6 and math.isfinite(r['predicted_pt']);groups[r['episode_id']].append(r)
assert len(groups)==7
per=[]
for eid,rr in sorted(groups.items()):
 y=np.array([r['target_pt'] for r in rr]);p=np.array([r['predicted_pt'] for r in rr]);corr=float(np.corrcoef(y,p)[0,1]) if len(rr)>1 and np.ptp(y)>0 and np.ptp(p)>0 else None
 per.append({'episode_id':eid,'samples':len(rr),'mae_pp':float(np.abs(p-y).mean()*100),'pearson':corr})
valid=[p['pearson'] for p in per if p['pearson'] is not None];mae=sum(p['mae_pp']*p['samples'] for p in per)/1744;correlation=float(np.mean(valid)) if valid else None
saved=read(out/'evaluation/metrics.json');assert abs(saved['eval_mae_pp']-mae)<1e-4 and abs(saved['eval_episode_mean_pearson']-correlation)<1e-6
result={'status':'passed','checkpoint':report['final_checkpoint'],'additional_epochs':3,'initial_checkpoint':cfg['initial_checkpoint'],'samples':1744,'episodes':7,'mae_pp':mae,'pearson':correlation,'pearson_undefined_episodes':7-len(valid),'pearson_definition':'equally averaged per-episode Pearson','mae_definition':'frame-weighted absolute error in percentage points','per_episode':per}
(out/'evaluation/verified-metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(S/'checkpoint-verification.json').write_text(json.dumps({'status':'passed','archives':archives},indent=2)+'\n');print(json.dumps(result))
