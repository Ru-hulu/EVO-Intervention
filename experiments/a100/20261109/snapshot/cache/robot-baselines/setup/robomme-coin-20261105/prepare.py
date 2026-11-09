"""CPU preparation: immutable source, episode-block split, train-only statistics."""
import json, subprocess, concurrent.futures
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path('/cache/robot-baselines')
SOURCE=ROOT/'data/lerobot/State-往杯子里放硬币_9ea3a83e_20261104_191614'
OUT=ROOT/'data/robomme-state-coin-20261105'
CAMS=['right_environment_1','left_wrist','right_wrist']
MASK=np.array([True]*6+[False]+[True]*6+[False])
def write_json(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 info=json.loads((SOURCE/'meta/info.json').read_text())
 df=pd.concat([pd.read_parquet(p) for p in sorted((SOURCE/'data').rglob('*.parquet'))],ignore_index=True)
 eps=pd.concat([pd.read_parquet(p) for p in sorted((SOURCE/'meta/episodes').rglob('*.parquet'))]).sort_values('episode_index')
 assert len(eps)==100 and len(df)==65398
 ids=eps.episode_index.to_numpy(); rng=np.random.default_rng(42)
 # One complete block of five episodes from each chronological quarter.
 val=sorted(int(i) for q in range(4) for i in ids[q*25+int(rng.integers(5))*5:][:5])
 train=[int(i) for i in ids if i not in val]; assert len(val)==20 and not set(train)&set(val)
 split={'seed':42,'method':'5-consecutive-episode blocks, one randomly selected block in each chronological quarter','train':train,'val':val,'no_independent_test_set':True}
 write_json(OUT/'split.json',split)
 arrays={}; tasks=pd.read_parquet(SOURCE/'meta/tasks.parquet'); prompt=str(tasks.index[0])
 action_stats=[]; state_stats=[]; records=[]
 for _,e in eps.iterrows():
  eid=int(e.episode_index); d=df[df.episode_index==eid].sort_values('frame_index'); n=len(d)
  assert n==int(e.length) and np.array_equal(d.frame_index,np.arange(n))
  s=np.stack(d['observation.state']).astype('float32'); a=np.stack(d.action).astype('float32')
  assert np.isfinite(s).all() and np.isfinite(a).all()
  anchors=np.arange(0,n,3); epdir=OUT/f'episode_{eid:03d}'; epdir.mkdir(exist_ok=True)
  np.savez(epdir/'trajectory.npz',state=s,action=a,anchors=anchors)
  if not (epdir/'images.npy').exists():
   mm=np.lib.format.open_memmap(epdir/'images.npy',mode='w+',dtype='uint8',shape=(len(anchors),3,224,224,3)); del mm
  arrays[eid]=np.load(epdir/'images.npy',mmap_mode='r+')
  if eid in train:
   state_stats.append(s[anchors]); chunks=a[np.minimum(anchors[:,None]+np.arange(20),n-1)].copy()
   chunks[:,:,MASK]-=s[anchors,None,:][:,:,MASK]; action_stats.append(chunks.reshape(-1,14))
  records.append({'episode':eid,'frames':n,'anchors':len(anchors),'split':'train' if eid in train else 'val'})
 stats={}
 for name,items in [('state',state_stats),('actions',action_stats)]:
  a=np.concatenate(items); stats[name]={k:v.tolist() for k,v in {'mean':a.mean(0),'std':a.std(0),'q01':np.quantile(a,.01,axis=0),'q99':np.quantile(a,.99,axis=0)}.items()}
 write_json(OUT/'norm_stats.json',{'norm_stats':stats})
 write_json(OUT/'dataset.json',{'source':str(SOURCE),'prompt':prompt,'fps':30,'anchor_stride':3,'history_fps':10,'action_horizon':20,'action_dim':14,'camera_order':CAMS,'delta_mask':MASK.tolist(),'episodes':records,'feature_checkpoint':str(ROOT/'models/RoboMME/perceptual-framesamp-modul/79999')})
 def decode(camidx,cam):
  marker=OUT/f'decoded-{cam}.json'
  if marker.exists(): return json.loads(marker.read_text())
  prefix=f'videos/observation.images.{cam}'
  groups={}
  for _,e in eps.iterrows():
   path=SOURCE/info['video_path'].format(video_key=f'observation.images.{cam}',chunk_index=int(e[prefix+'/chunk_index']),file_index=int(e[prefix+'/file_index']))
   eid=int(e.episode_index); start=round(float(e[prefix+'/from_timestamp'])*30)
   for j,t in enumerate(range(0,int(e.length),3)):
    groups.setdefault(path,{})[start+t]=(eid,j)
  total=0
  for path,mapping in groups.items():
   cmd=['ffmpeg','-v','error','-threads','2','-i',str(path),'-vf','scale=224:126,pad=224:224:0:49:black','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-']
   proc=subprocess.Popen(cmd,stdout=subprocess.PIPE); index=0; found=0; size=224*224*3
   while True:
    raw=proc.stdout.read(size)
    if not raw: break
    assert len(raw)==size
    if index in mapping:
     eid,j=mapping[index]; arrays[eid][j,camidx]=np.frombuffer(raw,np.uint8).reshape(224,224,3); found+=1
    index+=1
   assert proc.wait()==0 and found==len(mapping),(path,found,len(mapping))
   total+=found; print('decoded',cam,path.name,found,flush=True)
  for mm in arrays.values(): mm.flush()
  report={'camera':cam,'frames':total}; write_json(marker,report); return report
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  decoded=list(pool.map(lambda x:decode(*x),enumerate(CAMS)))
 report={'train_episodes':len(train),'val_episodes':len(val),'train_anchors':sum(e['anchors'] for e in records if e['split']=='train'),'val_anchors':sum(e['anchors'] for e in records if e['split']=='val'),'decoded':decoded,'finite':True,'split_disjoint':True}
 write_json(OUT/'prepared.json',report); print(json.dumps(report),flush=True)
if __name__=='__main__': main()
