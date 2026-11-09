"""Two-GPU cached Forest evaluation, episode-equal Pearson and frame-equal MAE."""
from pathlib import Path
import argparse,json,os,sys,time,math
S=Path(__file__).resolve().parent
D=Path('/cache/cvpzhr/datasets/forest-flipcup-5fps-no-augmentation-20261109')
sys.path.insert(0,str(S/'code'))
def read(p):return json.loads(Path(p).read_text())
def save(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def groups(records,n=2):
 result=[[] for _ in range(n)];loads=[0]*n
 for r in sorted(records,key=lambda x:(-x['augmented_samples'],x['episode_id'])):
  i=min(range(n),key=loads.__getitem__);result[i].append(r);loads[i]+=r['augmented_samples']
 return result

def summarize(rows):
 import numpy as np
 by={}
 for r in rows:by.setdefault(r['episode_id'],[]).append(r)
 episodes=[]
 for eid,rr in sorted(by.items()):
  y=np.array([r['target_pt'] for r in rr],dtype=np.float64);p=np.array([r['predicted_pt'] for r in rr],dtype=np.float64)
  assert np.isfinite(y).all() and np.isfinite(p).all()
  centered_y=y-y.mean();centered_p=p-p.mean()
  # Exact constant sequences are undefined; do not replace their correlation with zero.
  undefined=len(rr)<2 or np.ptp(y)==0 or np.ptp(p)==0
  denom=np.linalg.norm(centered_y)*np.linalg.norm(centered_p)
  corr=None if undefined else float(np.clip(np.dot(centered_y,centered_p)/denom,-1,1))
  episodes.append({'episode_id':eid,'task':rr[0]['task'],'frames':len(rr),'mae':float(abs(p-y).mean()),'pearson':corr,'undefined_target_constant':bool(np.ptp(y)==0),'undefined_prediction_constant':bool(np.ptp(p)==0)})
 def agg(eps):
  defined=[e['pearson'] for e in eps if e['pearson'] is not None];frames=sum(e['frames'] for e in eps)
  mae=sum(e['mae']*e['frames'] for e in eps)/frames
  return {'episodes':len(eps),'frames':frames,'mae':mae,'mae_pp':100*mae,'pearson':sum(defined)/len(defined) if defined else None,'pearson_defined_episodes':len(defined),'pearson_undefined_episodes':len(eps)-len(defined),'constant_target_episodes':sum(e['undefined_target_constant'] for e in eps),'constant_prediction_episodes':sum(e['undefined_prediction_constant'] for e in eps)}
 return {'overall':agg(episodes),'by_task':{t:agg([e for e in episodes if e['task']==t]) for t in sorted({e['task'] for e in episodes})},'episodes':episodes}

class EvalCollator:
 def __init__(self,processor,dataset,config):
  from mem_lite.datasets.m3_dataset import Stage3Collator
  self.collator=Stage3Collator(processor,dataset.history_dim,config['max_sequence_length'],dataset.history_capacity)
 def __call__(self,rows):
  b=self.collator(rows);b['rows']=[{k:r[k] for k in ['episode_id','frame_index','target_pt']} for r in rows];return b

def worker(a):
 import torch
 from torch.utils.data import DataLoader,Subset
 from mem_lite.datasets.m3_dataset import M3Dataset
 from mem_lite.runtime import load_checkpoint,to_device
 rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2)
 device=f'cuda:{rank}';out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 if not a.smoke:assert checkpoint['epoch']==3 and checkpoint['next_batch']==0
 model,processor,config=load_checkpoint(a.checkpoint,device);del checkpoint
 records=[r for r in read(D/'plan.json')['records'] if r['split']=='validation'];records=groups(records)[rank]
 ds=M3Dataset([r['output_dir'] for r in records]);mapping={r['output_dir']:r for r in records};indices=None
 assert ds.history_source==config['history_source'] and ds.history_capacity==config['history_capacity'] and ds.history_dim==config['model']['history_dim']
 if a.smoke:
  # First and last observations exercise both empty and full history for every episode.
  indices=sorted(set([i for i,(episode,row,_) in enumerate(ds.samples) if i==0 or i==len(ds)-1 or (i and ds.samples[i-1][0]!=episode) or (i+1<len(ds) and ds.samples[i+1][0]!=episode)]))
 loader=DataLoader(Subset(ds,indices) if indices is not None else ds,batch_size=a.batch_size,num_workers=a.workers,pin_memory=True,collate_fn=EvalCollator(processor,ds,config),multiprocessing_context='spawn' if a.workers else None)
 path=out/f'predictions-rank{rank}.jsonl';tmp=path.with_suffix('.jsonl.tmp');count=0;started=time.time()
 with torch.inference_mode(),tmp.open('w') as f:
  for batch in loader:
   pred=model(to_device(batch['inputs'],device))['pt'].float().cpu().tolist()
   assert len(pred)==len(batch['rows'])
   for r,p in zip(batch['rows'],pred):
    assert math.isfinite(p);source=mapping[r['episode_id']]
    row=dict(r,episode_id=source['episode_id'],task=source['task'],predicted_pt=p)
    f.write(json.dumps(row,allow_nan=False)+'\n');count+=1
   f.flush();save(out/f'worker-{rank}.json',{'phase':'running','rank':rank,'pid':os.getpid(),'samples':count,'updated_at':time.time()})
 tmp.replace(path);save(out/f'worker-{rank}.json',{'phase':'completed','rank':rank,'samples':count,'seconds':time.time()-started})

def aggregate(a):
 records=[r for r in read(D/'plan.json')['records'] if r['split']=='validation'];expected={}
 for r in records:
  for line in (Path(r['output_dir'])/'samples.jsonl').read_text().splitlines():
   x=json.loads(line);expected[(r['episode_id'],x['frame_index'])]=x['pt']
 rows=[];seen=set()
 for rank in range(2):
  for line in (Path(a.output)/f'predictions-rank{rank}.jsonl').read_text().splitlines():
   r=json.loads(line);k=(r['episode_id'],r['frame_index']);assert k not in seen and k in expected;seen.add(k)
   assert abs(expected[k]-r['target_pt'])<1e-12;rows.append(r)
 if not a.smoke:assert seen==set(expected) and len(rows)==1744 and len(records)==7
 metrics=summarize(rows);metrics.update(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_epoch=3 if not a.smoke else None,validation_augmented=False,coverage_complete=seen==set(expected),protocol={'pearson':'per episode, equally averaged over defined episodes; constants reported as undefined','mae':'all frames equally weighted; mae_pp=100*mae','tasks':1,'seed':42})
 save(Path(a.output)/'metrics.json',metrics)
 print(json.dumps({'overall':metrics['overall'],'by_task':metrics['by_task']},ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=4);p.add_argument('--aggregate',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.aggregate:aggregate(a)
 else:worker(a)
