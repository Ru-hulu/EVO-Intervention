from pathlib import Path
import json
import random
import torch
from transformers import AutoProcessor
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')
config=json.loads(Path('/cache/cvpzhr/setup/proprimem-5fps-20261030/train-4gpu.template.json').read_text())
torch.set_num_threads(4)
dataset=M3Dataset(config['sample_files'])
processor=AutoProcessor.from_pretrained(config['model_path'],max_pixels=config['max_pixels'],local_files_only=True)
collator=Stage3Collator(processor,dataset.history_dim,config['max_sequence_length'],dataset.history_capacity)
last={}
for i,(episode,row,previous) in enumerate(dataset.samples):last[episode]=i
candidates=[]
for episode,i in last.items():
    row=dataset.samples[i][1];directory,summary=dataset.episodes[episode]
    size=(directory/row['history']).stat().st_size if row['history'] else 0
    # 历史容量、文本长度及真实视角数量共同决定最长的训练输入。
    score=size/5120+len(processor.tokenizer.encode(summary['task_prompt']))+sum(v is not None for v in row['images'].values())*64
    candidates.append((score,i))
long=[]
for _,i in sorted(candidates,reverse=True)[:64]:
    sample=dataset[i];batch=collator([sample])['inputs']
    long.append(dict(index=i,sequence_length=int(batch['attention_mask'].sum()),
                     history_length=int(batch['history_valid_mask'].sum()),episode=sample['episode_id']))
long.sort(key=lambda r:r['sequence_length'],reverse=True)
result=dict(dataset_signature=dataset.signature,total_samples=len(dataset),
            random_indices=random.Random(42).sample(range(len(dataset)),8192),long=long,
            stress_index=long[0]['index'],stress_sequence_length=long[0]['sequence_length'])
(RUN/'selection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(total_samples=len(dataset),stress=long[0],min_candidate_length=long[-1]['sequence_length']),ensure_ascii=False))
