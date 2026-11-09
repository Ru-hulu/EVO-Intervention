"""真实 Qwen3.5-4B，双/三视角缓存与在线前向校验；不更新参数。"""
import csv
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoProcessor

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.datasets.prepare_proprimem import label_source, write_json
from mem_lite.datasets.prepare_stream import DEFAULT_MEMORY_CONFIG, build_episode, iter_observations
from mem_lite.datasets.views import VIEW_POLICY
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.runtime import StreamingProgressPredictor, to_device

output = Path(sys.argv[1])
model_path = '/cache/cvpzhr/models/Qwen3.5-4B'
index = json.loads((output/'index.json').read_text())
sources = [min((r for r in index['sources'] if r['pt_method'].startswith('whole-video')
                and sum(v is not None for v in r['videos']) == count), key=lambda r:r['frames']) for count in (2,3)]
torch.set_num_threads(4)
started = perf_counter()
encoder = load_vision_encoder(model_path,'cuda')
visual, processor, vision_hash = encoder
config = json.loads(DEFAULT_MEMORY_CONFIG.read_text())
origin = dict(method='streamforest-v1', vision_sha256=vision_hash, memory_config=config,
              sample_fps=5.0, include_episode_last_frame=True, max_pixels=65536, min_pixels=4096, view_policy=VIEW_POLICY)
stream = StreamMemory(QwenFrameEncoder(visual), processor, config, 65536, 4096)
directories = []
for row in sources:
    units = label_source(row, output, Path(index['legacy_dir']), encoder, 16, 262144)
    unit = units[0]
    directory = output/'cache'/row['split']/unit['unit_id']
    with Path(unit['progress_csv']).open() as f:
        progress = {int(r['frame_index']):float(r['pt']) for r in csv.DictReader(f)}
    observations = iter_observations(row['videos'],progress, sample_fps=5.0,
                                    start_frame=0,end_frame=unit['frame_range'][1])
    build_episode(observations,stream,directory,row['task_prompt'],unit['unit_id'],origin,source={**row,**unit})
    directories.append(directory)
del stream,encoder,visual,processor
gc.collect()
torch.cuda.empty_cache()
model = M3ProgressPredictor(model_path, dtype=torch.bfloat16, lora_dropout=0, gradient_checkpointing=False).to('cuda').eval()
processor = AutoProcessor.from_pretrained(model_path,max_pixels=65536,local_files_only=True)
results = []
for directory, row in zip(directories,sources):
    data = M3Dataset([directory])
    collator = Stage3Collator(processor,data.history_dim,history_capacity=data.history_capacity)
    online = StreamingProgressPredictor(model,processor,origin)
    errors = []
    for i in range(min(4,len(data))):
        sample = data[i]
        actual_views = [img if row['videos'][j] is not None else None for j,img in enumerate(sample['images'])]
        with torch.no_grad():
            cached = model(to_device(collator([sample])['inputs'],'cuda'))['pt']
            predicted = online(actual_views,sample['frame_index'],sample['episode_id'],sample['task_prompt'])['pt']
        assert torch.isfinite(cached).all() and torch.isfinite(predicted).all()
        torch.testing.assert_close(cached,predicted,atol=1e-5,rtol=1e-5)
        errors.append(float((cached-predicted).abs().max()))
    results.append(dict(source=row['source_id'],views=sum(v is not None for v in row['videos']),
                        samples=len(data),max_prediction_difference=max(errors)))
write_json(output/'gpu-smoke.json',dict(status='passed',results=results,parameter_updates=0,
           elapsed_seconds=perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
print(json.dumps(results),flush=True)
