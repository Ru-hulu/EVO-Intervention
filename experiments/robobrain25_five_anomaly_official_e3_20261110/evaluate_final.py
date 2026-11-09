"""Target-free native GRM generation from epoch3, two disjoint episode shards, resumable output."""
import argparse,gc,json,math,os,time
from pathlib import Path
from collections import defaultdict
import torch
from transformers import set_seed
from grm import Dataset,Collator,load_model,load_processor,generate,parse_score
from common import *
from metrics import summarize,correlation

def verify(c):
 out=Path(c['output_dir']);archives=[]
 from safetensors.torch import load_file
 for epoch in range(1,4):
  step=epoch*c['steps_per_epoch'];d=out/f'checkpoint-{step}';receipt=read(d/'complete.json');state=read(d/'trainer_state.json')
  assert receipt['complete'] and receipt['saved_before_validation'] and state['global_step']==step and math.isclose(state['epoch'],epoch,abs_tol=1e-8)
  assert read(d/'run_config.json')==c
  hashes={}
  for n in receipt['files']:
   p=d/n;assert p.is_file() and p.stat().st_size>0;hashes[n]=sha(p)
  assert all(torch.isfinite(x).all() for x in load_file(d/'adapter_model.safetensors').values())
  archives.append(dict(epoch=epoch,step=step,checkpoint=str(d),sha256=hashes))
 audits=[read(Path(c['setup_dir'])/f'initialization-rank{i}.json') for i in [0,1]]
 assert audits[0]['trainable_sha256']==audits[1]['trainable_sha256']
 assert all(a['fresh_lora_B_all_zero'] and a['initial_checkpoint'] is None and a['only_language_lora_trainable'] and a['official_revision']==c['model_revision'] for a in audits)
 save(Path(c['setup_dir'])/'checkpoint-verification.json',dict(status='passed',archives=archives))
 return archives[-1]

def worker(c):
 rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));set_seed(c['seed']);d=Path(c['output_dir'])/'evaluation';d.mkdir(exist_ok=True)
 ck=Path(c['output_dir'])/f"checkpoint-{c['total_steps']}";assert read(ck/'complete.json')['step']==c['total_steps']
 ds=Dataset(c['data_dir'],'validation');epids=sorted(ds.episodes)
 ids=sorted({r['episode_id'] for r in ds.rows});assignment={eid:i%2 for i,eid in enumerate(ids)}
 indices=[i for i,r in enumerate(ds.rows) if assignment[r['episode_id']]==rank]
 path=d/f'predictions-rank{rank}.jsonl';existing=rows(path) if path.exists() else [];seen={}
 for x in existing:
  key=(x['episode_id'],x['sample_index']);assert key not in seen;seen[key]=x
 for i in indices:
  row=ds.rows[i];key=(row['episode_id'],row['sample_index'])
  if key in seen:assert seen[key]['target_pt']==row['pt'] and seen[key]['source_frame_index']==row['source_frame_index']
 assert set(seen)<={(ds.rows[i]['episode_id'],ds.rows[i]['sample_index']) for i in indices}
 indices=[i for i in indices if (ds.rows[i]['episode_id'],ds.rows[i]['sample_index']) not in seen]
 metadata=d/'evaluation-source.json';source=dict(checkpoint=str(ck),adapter_sha256=sha(ck/'adapter_model.safetensors'),config_sha256=sha(Path(c['setup_dir'])/'execution.config.json'))
 if metadata.exists():assert read(metadata)==source
 elif rank==0:save(metadata,source)
 proc=load_processor(c['model_path'],c['min_pixels'],c['max_pixels']);model=load_model(c['model_path'],adapter=ck).to('cuda').eval();collator=Collator(proc,c['max_length'])
 batchsize=9;pos=0;invalid=sum(x['predicted_pt'] is None for x in existing);start=time.time()
 with path.open('a') as f:
  while pos<len(indices):
   selected=[ds[i] for i in indices[pos:pos+batchsize]];batch=None
   try:
    batch={k:v.to('cuda') for k,v in collator.encode(selected,supervised=False).items()};assert 'labels' not in batch and 'target_pt' not in batch
    answers=generate(model,proc,batch,c['max_new_tokens'])
   except torch.cuda.OutOfMemoryError:
    batch=None;gc.collect();torch.cuda.empty_cache()
    if batchsize==1:raise
    batchsize=max(1,batchsize//2);continue
   for row,text in zip(selected,answers,strict=True):
    record={k:row[k] for k in ['episode_id','sample_index','source_frame_index','timestamp_seconds','kind']};record.update(target_pt=row['pt'],raw_response=text)
    try:record['predicted_pt']=parse_score(text)
    except ValueError as e:record.update(predicted_pt=None,error=str(e));invalid+=1
    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
   f.flush();pos+=len(selected);del batch
   save(d/f'worker-{rank}.json',dict(phase='evaluating',samples=len(existing)+pos,total=len(existing)+len(indices),invalid=invalid,seconds=time.time()-start))
 save(d/f'worker-{rank}.json',dict(phase='completed',samples=len(existing)+pos,invalid=invalid,seconds=time.time()-start))

def aggregate(c):
 out=Path(c['output_dir']);d=out/'evaluation';expected=rows(Path(c['data_dir'])/'validation.jsonl');lookup={(r['episode_id'],r['sample_index']):r for r in expected};actual={}
 for rank in [0,1]:
  assert read(d/f'worker-{rank}.json')['phase']=='completed'
  for r in rows(d/f'predictions-rank{rank}.jsonl'):
   key=(r['episode_id'],r['sample_index']);assert key not in actual and key in lookup;ref=lookup[key]
   assert r['target_pt']==ref['pt'] and r['source_frame_index']==ref['source_frame_index'];actual[key]=r
 assert set(actual)==set(lookup) and len(actual)==c['counts']['validation']['samples']
 rr=[actual[k] for k in sorted(actual)];invalid=[r for r in rr if r['predicted_pt'] is None]
 result=dict(status='completed' if not invalid else 'invalid_generations',checkpoint=str(out/f"checkpoint-{c['total_steps']}"),checkpoint_epoch=3,
  coverage_complete=not invalid,samples=len(rr),episodes=len({r['episode_id'] for r in rr}),invalid_generations=len(invalid),uses_future_goal_reference=True,
  protocol={'MAE':'frame equal','Pearson':'episode equal; constants undefined'},validation_augmented=False)
 with (d/'predictions.jsonl').open('w') as f:
  for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 if not invalid:
  result.update(summarize(rr));by=defaultdict(list)
  for r in rr:by[r['episode_id']].append(r)
  result['by_episode']=[dict(episode_id=e,samples=len(v),mae_pp=100*sum(abs(r['predicted_pt']-r['target_pt']) for r in v)/len(v),
   pearson=correlation([r['predicted_pt'] for r in v],[r['target_pt'] for r in v])) for e,v in sorted(by.items())]
 save(d/'metrics.json',result)
 if invalid:raise RuntimeError(f'{len(invalid)} invalid native GRM answers; raw predictions retained, full metrics withheld')
 print(json.dumps(result['overall']),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--verify',action='store_true');p.add_argument('--aggregate',action='store_true');a=p.parse_args();c=read(a.config)
 if a.verify:verify(c)
 elif a.aggregate:aggregate(c)
 else:worker(c)
