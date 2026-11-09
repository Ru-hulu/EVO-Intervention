import json
from pathlib import Path
import torch
from transformers import AutoProcessor
from mem_lite.datasets.m3_dataset import M3Dataset,Stage3Collator
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.runtime import StreamingProgressPredictor,to_device

RUN=Path('/cache/cvpzhr/runs/forest-proprimem-e3-20261031-0009')
config=json.loads(Path('/cache/cvpzhr/setup/proprimem-5fps-20261030/train-4gpu.template.json').read_text())
dataset=M3Dataset(config['sample_files'])
torch.manual_seed(42);torch.set_num_threads(4)
model=M3ProgressPredictor(config['model_path'],**{**config['model'],'gradient_checkpointing':False,'lora_dropout':0}).to('cuda').eval()
processor=AutoProcessor.from_pretrained(config['model_path'],max_pixels=config['max_pixels'],local_files_only=True)
collator=Stage3Collator(processor,dataset.history_dim,config['max_sequence_length'],dataset.history_capacity)
results=[]
with torch.no_grad():
    for count in (2,3):
        start=next(i for i,(_,sample,previous) in enumerate(dataset.samples) if previous is None and sum(v is not None for v in sample['images'].values())==count)
        online=StreamingProgressPredictor(model,processor,dataset.history_source)
        for i in range(start,start+3):
            sample=dataset[i];inputs=to_device(collator([sample])['inputs'],'cuda')
            cached=model(inputs)['pt'];streamed=online(sample['images'],sample['frame_index'],sample['episode_id'],sample['task_prompt'])['pt']
            torch.testing.assert_close(cached,streamed,atol=1e-5,rtol=1e-5)
            results.append(dict(views=count,frame_index=sample['frame_index'],cache_online_difference=float((cached-streamed).abs().max())))
    selection=json.loads((RUN/'selection.json').read_text());sample=dataset[selection['stress_index']]
    padded=to_device(collator([sample])['inputs'],'cuda');keep=padded['attention_mask'].any(0)
    compact={k:(v[:,keep] if k in ('input_ids','attention_mask','mm_token_type_ids','history_mask') else v) for k,v in padded.items()}
    left=model(padded)['pt'];right=model(compact)['pt'];difference=float((left-right).abs().max())
    torch.testing.assert_close(left,right,atol=1e-3,rtol=0)
report=dict(status='passed',parameter_updates=0,cache_online=results,padding_max_difference=difference,
            padded_length=padded['input_ids'].shape[1],valid_length=compact['input_ids'].shape[1])
(RUN/'padding-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
