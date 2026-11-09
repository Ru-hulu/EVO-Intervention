from pathlib import Path
import json,torch
from robometer_training.data import ProgressCollator,ProgressDataset
from robometer_training.model import load_processor
ROOT=Path('/cache/cvpzhr');RUN=ROOT/'setup/robometer-20261031';DATA=ROOT/'datasets/robometer-proprimem-5fps-w8-20261031'
torch.set_num_threads(2)
p=load_processor(ROOT/'models/Robometer-4B',min_pixels=4096,max_pixels=65536);c=ProgressCollator(p)
selected={}
for row in ProgressDataset(DATA/'validation.jsonl').rows:
 task=row['source_id'].rsplit('/',1)[0]
 if len(set(row['window_indices']))==8:selected[task]=row
rows=list(selected.values())
for start in range(0,len(rows),4):
 batch=rows[start:start+4];cached=c.encode(batch,supervised=True);online=c.encode([{k:v for k,v in r.items() if k not in ('pixel_cache','window_indices')} for r in batch],supervised=True)
 assert cached.keys()==online.keys()
 for k in cached:assert torch.equal(cached[k],online[k].to(cached[k].dtype)),(k,[r['episode_id'] for r in batch])
report=dict(status='passed',tasks=len(rows),samples=len(rows),mixed_task_padding_equal=True,all_input_tensors_equal=True,pixel_comparison_dtype='bfloat16',labels_equal=True)
(RUN/'mixed-input-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
