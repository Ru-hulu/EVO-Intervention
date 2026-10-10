import gc, os, json, random, time, ctypes
from pathlib import Path
from qwen_training.dataset import VLMProgressDataset

def mem(label):
 fields=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
 print(json.dumps(dict(label=label,rss=fields['VmRSS'].strip(),threads=fields['Threads'].strip(),gc_count=gc.get_count(),time=time.time())),flush=True)

if __name__=='__main__':
 mem('imports')
 d=VLMProgressDataset(['/root/dataset/RW-RL-Dataset/piperx_insert_copper_screw/samples/samples.jsonl','/root/dataset/piperx-bolt-intentional-drop-reinsert-22ep-raw/samples/samples.jsonl'])
 mem('index')
 rng=random.Random(1)
 for i in range(1000):
  x=d[rng.randrange(len(d))]
  del x
  if (i+1)%100==0: mem(f'decoded-{i+1}')
  if (i+1)%250==0:
   print('GC',gc.collect(),flush=True);mem('after-gc')
   ctypes.CDLL(None).malloc_trim(0);mem('after-trim')
