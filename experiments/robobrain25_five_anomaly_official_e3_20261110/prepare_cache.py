"""Independent clean GRM cache; original pixels and PT, target-free generation checks."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,csv,hashlib,json,os,shutil,time
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
from common import *

def check_episode(job):
 import torch
 torch.set_num_threads(1)
 from grm import Dataset,Collator,load_processor,pixel_cache,parse_score
 directory,eid=job;ds=Dataset(directory,'train');ds.rows += rows(Path(directory)/'validation.jsonl')
 ids=[i for i,r in enumerate(ds.rows) if r['episode_id']==eid]
 proc=load_processor('/cache/cvpzhr/models/RoboBrain2.5-4B');collator=Collator(proc)
 checked=[];maximum=0
 for i in sorted({ids[0],ids[len(ids)//2],ids[-1]}):
  row=ds[i]
  for supervised in [True,False]:
   a=collator.encode([row],supervised=supervised,cached=True)
   b=collator.encode([row],supervised=supervised,cached=False)
   assert a.keys()==b.keys()
   for key in a:
    aa,bb=a[key],b[key]
    if key=='pixel_values':aa,bb=aa.to(torch.bfloat16),bb.to(torch.bfloat16)
    assert torch.equal(aa,bb),(eid,i,key)
   maximum=max(maximum,a['input_ids'].shape[1])
   if supervised:
    assert int((a['labels']!=-100).sum())>0
    assert all(not x.startswith('target') for x in b if not supervised)
   else:
    assert 'labels' not in a and 'target_pt' not in a
    altered=dict(row,pt=1-row['pt'],answer='<score>+99.9%</score>')
    aa=collator.encode([altered],supervised=False)
    assert all(torch.equal(a[k],aa[k]) for k in a)
  checked.append(i)
 pixel_cache.cache_clear()
 return {'episode_id':eid,'examples':len(checked),'max_tokens':maximum,'cached_equals_official_processor':True,'target_independent_inference':True}

def prepare(task):
 from grm import answer,parse_score
 c=config(task);d=Path(c['data_dir']);assert not d.exists(),d;d.mkdir(parents=True)
 rc=read(f'/cache/cvpzhr/setup/forest-robometer-{task}-official-e3-20261109/robometer/training.config.json')
 rm=read(Path(rc['train_data']).parent/'manifest.json');pixelinfo={e['episode_id']:e for sp in rm['splits'].values() for e in sp['episodes']}
 episodes={};counts={};protected={};allids={};quant=0
 for split in ['train','validation']:
  sources=read(Path(rc['cache_setup_dir'])/'source'/f'{split}.sample_files.json')
  expectedrows=rows(rc['train_data'] if split=='train' else rc['validation_data']);lookup={(r['episode_id'],r['source_frame_index']):r for r in expectedrows}
  seen=set();ids=set();n=0
  with (d/f'{split}.jsonl').open('w') as out:
   for folder in sources:
    source=Path(folder);summary=read(source/'summary.json');rr=rows(source/'samples.jsonl');eid=summary['episode_id'];original=summary['source']
    ids.add(eid);assert eid not in episodes and original['task']==c['task_id']
    assert rr[0]['frame_index']==0 and rr[0]['pt']==0 and rr[-1]['frame_index']==original['frames']-1 and rr[-1]['pt']==1
    with Path(original['progress_csv']).open() as f:labels={int(x['frame_index']):float(x['pt']) for x in csv.DictReader(f)}
    assert sha(original['progress_csv'])==original['progress_sha256']
    protected[str(source/'samples.jsonl')]=sha(source/'samples.jsonl');protected[str(source/'summary.json')]=sha(source/'summary.json')
    destination=d/'images'/eid;images=[]
    for row in rr:
     entry=[]
     for view in ['cam_high','cam_left_wrist','cam_right_wrist']:
      rel=row['images'][view];assert rel and not Path(rel).is_absolute() and '..' not in Path(rel).parts
      target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True)
      if not target.exists():os.link(source/rel,target)
      assert target.samefile(source/rel);entry.append(rel)
     images.append(entry)
    pixel=d/'pixels'/f'{eid}.safetensors';pixel.parent.mkdir(parents=True,exist_ok=True);os.link(pixelinfo[eid]['pixel_cache'],pixel)
    assert pixel.stat().st_size>0
    episodes[eid]=dict(source_dir=str(destination),task_prompt=summary['task_prompt'],suite=eid.split('/')[0],task_id=c['task_id'],
       native_images=images,pixel_cache=str(pixel),end_pixel_index=len(rr)-1,endpoint_native_frames=[0,rr[-1]['frame_index']],
       native_samples=len(rr),same_episode_final_frame_reference=True,split=split,original_source_dir=str(source),original_progress_csv=original['progress_csv'],fps=original['fps'])
    for i,row in enumerate(rr):
     assert row.get('kind','original')=='original' and row['pt']==labels[row['frame_index']]
     key=(eid,row['frame_index']);assert key in lookup and lookup[key]['pt']==row['pt'] and key not in seen;seen.add(key)
     assert lookup[key]['sample_index']==i and lookup[key]['source_sample_index']==i
     record=dict(episode_id=eid,sample_index=i,source_sample_index=i,anchor_sample_index=i,frame_index=row['frame_index'],source_frame_index=row['frame_index'],
        timestamp_seconds=row['frame_index']/original['fps'],kind='original',pt=row['pt'],answer=answer(row['pt']),pixel_index=i)
     quant=max(quant,abs(parse_score(record['answer'])-row['pt']));out.write(json.dumps(record,ensure_ascii=False)+'\n');n+=1
  assert seen==set(lookup) and n==c['counts'][split]['samples'] and len(ids)==c['counts'][split]['episodes']
  allids[split]=ids;counts[split]={'episodes':len(ids),'samples':n}
 assert allids['train'].isdisjoint(allids['validation']) and quant<=0.0005000001
 save(d/'episodes.json',episodes);save(d/'source-hashes.json',protected)
 save(d/'manifest.json',dict(status='prepared',counts=counts,input_contract=c['input_contract'],uses_future_goal_reference=True,augmentation='none',
    original_PT_preserved=True,quantization_max_pt=quant,pixel_cache='independent immutable hardlinks, verified against official processor',forest_features_used=False))
 return [(str(d),eid) for eid in episodes]

def main():
 start=time.time();save(ROOT/'cache-status.json',dict(phase='preparing',pid=os.getpid(),started_at=start));jobs=[]
 for task in TASKS:jobs.extend(prepare(task))
 reports={}
 with ProcessPoolExecutor(max_workers=8) as pool:
  futures={pool.submit(check_episode,j):j for j in jobs}
  for f in as_completed(futures):
   d,eid=futures[f];reports[eid]=f.result();save(ROOT/'cache-status.json',dict(phase='checking_processor_parity',episodes=len(reports),total_episodes=len(jobs),pid=os.getpid(),started_at=start))
 for task in TASKS:
  d=Path(config(task)['data_dir']);episodes=read(d/'episodes.json');rr=[reports[e] for e in episodes]
  assert all(sha(p)==h for p,h in read(d/'source-hashes.json').items())
  m=read(d/'manifest.json');m.update(status='completed',checks=rr,max_input_tokens=max(x['max_tokens'] for x in rr),input_files_sha256={n:sha(d/n) for n in ['episodes.json','train.jsonl','validation.jsonl']});save(d/'manifest.json',m)
 save(ROOT/'cache-status.json',dict(phase='completed',episodes=len(reports),total_episodes=len(jobs),pid=os.getpid(),seconds=time.time()-start))
 print('ALL_FIVE_CACHES_VERIFIED',len(reports),flush=True)
if __name__=='__main__':main()
