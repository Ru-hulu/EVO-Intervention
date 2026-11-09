"""Real-model preflight only; no parameter updates in this phase."""
import gc
import json
from pathlib import Path
import sys
import time

import torch
from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.m3_inputs import M3Collator
from mem_lite.runtime import StreamingProgressPredictor, load_checkpoint, to_device
from mem_lite.train_stage3 import load_trainable_state, trainable_state

output = Path(sys.argv[1])
dataset_root = Path('/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030')
model, processor, config = load_checkpoint(output/'last.pt')
state = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
assert state['step'] == 12
saved = state['model']
loaded = trainable_state(model)
for part in saved:
    assert saved[part].keys() == loaded[part].keys()
    for name in saved[part]:
        torch.testing.assert_close(loaded[part][name].cpu(), saved[part][name], atol=0, rtol=0)
gates = {name:float(module.gate) for name,module in model.memory_adapters.items()}
assert all(torch.isfinite(module.gate) and module.gate.abs() > 0 for module in model.memory_adapters.values())
report = {'checkpoint_step':state['step'], 'checkpoint_tensor_match':True, 'gates':gates}
del state, loaded

paths = {split:json.loads((dataset_root/f'{split}.sample_files.json').read_text()) for split in ('train','validation')}
train = M3Dataset(paths['train'])
validation = M3Dataset(paths['validation'])
source_ids = {split:{summary['source']['source_id'] for _,summary in data.episodes}
              for split,data in [('train',train),('validation',validation)]}
assert not source_ids['train'] & source_ids['validation']
assert train.history_source == validation.history_source == config['history_source']
report['dataset'] = {'train_samples':len(train),'validation_samples':len(validation),
                     'source_videos':{key:len(value) for key,value in source_ids.items()},'source_overlap':0}
collator = M3Collator(processor, train.history_dim, config['max_sequence_length'], train.history_capacity)

# Real long histories, choosing the final observation of each of the longest training units.
ends = {episode:index for index,(episode,_,_) in enumerate(train.samples)}
long_indices = [ends[episode] for episode in sorted(ends, key=lambda e:train.episodes[e][1]['num_samples'], reverse=True)[:16]]
long_samples = [train[index] for index in long_indices]
long_samples.sort(key=lambda row:len(row['history_tokens']), reverse=True)
long_samples = long_samples[:4]
inputs = to_device(collator(long_samples), 'cuda')
target = torch.tensor([s['target_pt'] for s in long_samples], device='cuda')
model.train()
model.zero_grad(set_to_none=True)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize(); started=time.perf_counter()
result = model(inputs,target)
result['loss'].backward()
torch.cuda.synchronize()
assert torch.isfinite(result['loss'])
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
report['long_history_backward'] = {
    'batch_size':4, 'tokens_per_sample':inputs['history_valid_mask'].sum(-1).tolist(),
    'main_sequence_length':inputs['input_ids'].shape[1], 'loss':float(result['loss']),
    'seconds':time.perf_counter()-started,
    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
    'memory_gradients_finite_and_nonzero':all(m.k_proj.weight.grad is not None and torch.isfinite(m.k_proj.weight.grad).all()
                                           and m.k_proj.weight.grad.abs().sum()>0 for m in model.memory_adapters.values()),
}
assert report['long_history_backward']['memory_gradients_finite_and_nonzero']
report['long_history_backward']['memory_gradients_finite_and_nonzero'] = True
model.zero_grad(set_to_none=True)
del result,inputs,target
gc.collect(); torch.cuda.empty_cache()
model.eval()

# Test causally cached and online histories on real two-view and three-view videos.
chosen = {}
for directory,summary in validation.episodes:
    views=sum(p is not None for p in summary['source']['videos'])
    if views in (2,3) and views not in chosen:
        chosen[views]=directory
assert set(chosen)=={2,3}
checks=[]
stream=StreamingProgressPredictor(model,processor,config['history_source'],config['max_sequence_length'])
with torch.no_grad():
    for views,directory in sorted(chosen.items()):
        data=M3Dataset([str(directory)])
        rows=[data[i] for i in range(min(16,len(data)))]
        stream.reset()
        differences=[]
        for row in rows:
            cached=model(to_device(collator([row]),'cuda'))['pt']
            online=stream(row['images'],row['frame_index'],row['episode_id'],row['task_prompt'])
            differences.append(float((cached-online['pt']).abs().max()))
        assert max(differences) <= 1e-5, differences
        reset=stream(rows[0]['images'],rows[0]['frame_index'],rows[0]['episode_id']+'-new',rows[0]['task_prompt'])
        first=model(to_device(collator([rows[0]]),'cuda'))['pt']
        assert reset['history_length']==0
        torch.testing.assert_close(reset['pt'],first,atol=1e-5,rtol=0)
        mixed=model(to_device(collator([rows[0],rows[-1]]),'cuda'))['pt']
        individual=torch.cat([model(to_device(collator([r]),'cuda'))['pt'] for r in (rows[0],rows[-1])])
        assert torch.isfinite(mixed).all()
        checks.append({'views':views,'episode':str(directory),'samples':len(rows),
                       'cached_online_max_difference':max(differences),'episode_reset':True,
                       'mixed_empty_history_batch_max_difference':float((mixed-individual).abs().max())})
    inputs=to_device(collator([rows[-1]]),'cuda')
    expected=model(inputs)['pt'].clone()
    model.progress_head.weight.add_(1)
    for module in model.memory_adapters.values():module.gate.add_(1)
    load_trainable_state(model,saved)
    torch.testing.assert_close(model(inputs)['pt'],expected,atol=0,rtol=0)
report.update(online_checks=checks,checkpoint_restoration_prediction_equal=True,status='passed',parameter_updates_in_this_phase=0)
(output/'preflight-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
