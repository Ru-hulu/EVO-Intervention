import gc,importlib.util,json,random
from pathlib import Path
import torch
from qwen_training.dataset import VLMProgressDataset
spec=importlib.util.spec_from_file_location('original_dataset','/data/evo-inter/code/qwen_training/dataset.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
paths=[('/root/dataset/RW-RL-Dataset/piperx_insert_copper_screw/samples/samples.jsonl',118990),('/root/dataset/piperx-bolt-intentional-drop-reinsert-22ep-raw/samples/samples.jsonl',80463)]
records=[]
for path,count in paths:
 chosen=set([0,1,count-1]+random.Random(42).sample(range(count),7))
 with open(path) as f:
  for i,line in enumerate(f):
   if i in chosen: records.append(json.loads(line))
a=old.VLMProgressDataset.__new__(old.VLMProgressDataset);a.samples=records
b=VLMProgressDataset.__new__(VLMProgressDataset);b.samples=records
for i in range(len(records)):
 x,y=a[i],b[i]
 for key in ('images','frame_mask'): assert torch.equal(x[key],y[key]),(i,key)
 for key in ('sample_id','input_state','target'): assert x[key]==y[key],(i,key)
 del x,y
 gc.collect()
 print('EXACT_MATCH',records[i]['sample_id'],flush=True)
Path('/data/evo-inter/verification/memory_fix_20261010/real_pixel_parity.json').write_text(json.dumps({'status':'PASS','samples':len(records),'ids':[x['sample_id'] for x in records],'views_per_sample':3,'frames_per_view':6,'pixel_comparison':'exact'},indent=2))
