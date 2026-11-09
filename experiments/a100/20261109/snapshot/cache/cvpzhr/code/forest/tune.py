"""Isolated real-model throughput/memory probes. No checkpoint is saved."""
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import triton
triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.uint8))
from transformers import AutoProcessor

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.train_stage3 import configure_history


ROOT = Path('/root/autodl-tmp/streamforest-cache/rebuild-20261030')
JOB = Path('/root/autodl-tmp/streamforest-jobs/train-3ep-20261030')
torch.set_num_threads(4)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
config = json.loads((ROOT / 'train.config.json').read_text())
dataset = M3Dataset(config['sample_files'])
configure_history(config, dataset)
processor = AutoProcessor.from_pretrained(config['model_path'], max_pixels=config['max_pixels'], local_files_only=True)
collator = Stage3Collator(processor, dataset.history_dim, config['max_sequence_length'], dataset.history_capacity)
offset, candidates, tasks = 0, [], set()
for _, summary in dataset.episodes:
    source = summary['source']
    task = source['dataset'] if source['dataset'] != 'memlong' else source['source_video_id'].split('/')[0]
    if task not in tasks:
        sample = dataset[offset + summary['num_samples'] - 1]
        width = collator([sample])['inputs']['input_ids'].shape[1]
        candidates.append((width, sample))
        tasks.add(task)
    offset += summary['num_samples']
worst_width, worst = max(candidates, key=lambda item:item[0])
config['model']['gradient_checkpointing'] = False
model = M3ProgressPredictor(config['model_path'], **config['model']).to('cuda').train()
parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'], fused=True)
results = []
limit = torch.cuda.get_device_properties(0).total_memory * .90


def run_case(batch_size, checkpointing):
    if checkpointing:
        model.qwen.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    else:
        model.qwen.model.language_model.gradient_checkpointing_disable()
    batch = collator([worst] * batch_size)
    inputs = {k:v.to('cuda') if torch.is_tensor(v) else v for k,v in batch['inputs'].items()}
    target = batch['target_pt'].to('cuda')
    times = []
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model(inputs, target)
        assert torch.isfinite(output['loss'])
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if repeat:
            times.append(time.perf_counter()-start)
        del output
    peak = torch.cuda.max_memory_allocated()
    return {'batch_size': batch_size, 'gradient_checkpointing': checkpointing,
            'seconds': statistics.median(times), 'samples_per_second': batch_size/statistics.median(times),
            'peak_memory_gib': peak/1024**3, 'safe': peak < limit,
            'benchmark_sequence_length': worst_width, 'status': 'PASS'}


for checkpointing, sizes in [(False,[4,6]),(True,[8,16,24])]:
    for size in sizes:
        print(json.dumps({'phase':'probe','batch_size':size,'gradient_checkpointing':checkpointing}),flush=True)
        try:
            result = run_case(size, checkpointing)
        except torch.OutOfMemoryError:
            result = {'batch_size':size,'gradient_checkpointing':checkpointing,'status':'OOM','safe':False}
        finally:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result)
        (JOB/'tuning.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not result['safe']:
            break
best = max((r for r in results if r['safe']), key=lambda r:r['samples_per_second'])
config.update(epochs=3,max_steps=None,batch_size=best['batch_size'],gradient_accumulation_steps=1,
              num_workers=8,persistent_workers=True,fused_optimizer=True,keep_only_latest=True,
              log_every_steps=1,save_every_steps=100,save_every_epochs=1,
              output_dir='/root/autodl-tmp/Evo-Inter/outputs/mem_lite/streamforest_recovery_memlong_e3_20261030')
config['model']['gradient_checkpointing']=best['gradient_checkpointing']
(JOB/'train.config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
(JOB/'tuning-choice.json').write_text(json.dumps(best,indent=2)+'\n')
print(json.dumps({'phase':'selected',**best}),flush=True)
