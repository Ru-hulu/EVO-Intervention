from pathlib import Path
import json,sys,hashlib
S=Path(__file__).resolve().parent;D=Path('/cache/cvpzhr/datasets/forest-pencil-5fps-no-augmentation-20261109');sys.path.insert(0,str(S/'code'))
import torch;torch.set_num_threads(2)
from mem_lite.datasets.m3_dataset import M3Dataset
read=lambda p:json.loads(p.read_text());cfg=read(S/'training.config.prepared.json');plan=read(D/'plan.json');report={'status':'passed'}
for split,count,episodes in [('train',10789,28),('validation',2177,7)]:
 records=[r for r in plan['records'] if r['split']==split];paths=read(D/f'{split}.sample_files.json');assert paths==[r['output_dir'] for r in records]
 for r in records:
  p=Path(r['output_dir']);m=read(p/'summary.json');assert m['augmentation']['enabled'] is False and m['augmentation']['inserted_observations']==0
  receipt=read(D/'receipts'/f"{r['episode_id']}.json")
  for name,key in [('samples.jsonl','samples_sha256'),('summary.json','summary_sha256')]:assert hashlib.sha256((p/name).read_bytes()).hexdigest()==receipt[key]
 ds=M3Dataset(paths);assert len(ds)==count and len(ds.episodes)==episodes and ds.history_source==cfg['history_source']
 for idx in [0,len(ds)-1]:assert 0<=ds[idx]['target_pt']<=1
 report[f'{split}_signature']=ds.signature;report[split]={'samples':count,'episodes':episodes}
(S/'training-input-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
