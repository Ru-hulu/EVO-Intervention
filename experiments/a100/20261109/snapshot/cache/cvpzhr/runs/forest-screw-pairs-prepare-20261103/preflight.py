"""Real paired data: checkpoint warm start, recurrent chunks and one discarded optimizer step."""
import csv, json, os, pathlib, time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import torch
from transformers import AutoProcessor
from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.sequence import EpisodeWindows, SequenceCollator
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.train_stage3 import initialize_weights
from mem_lite.memory.vision import verify_vision_source
from mem_lite.runtime import to_device

run=pathlib.Path('/cache/cvpzhr/runs/forest-screw-pairs-prepare-20261103')
prepared=pathlib.Path('/cache/EVO-Intervention/datasets/insert-screw-recover-pairs-20261103')
cache=pathlib.Path('/cache/cvpzhr/datasets/forest-screw-pairs-5fps-20261103')
cfg=json.loads((run/'finetune-config.json').read_text())
rows=[json.loads(line) for line in (prepared/'train.jsonl').read_text().splitlines()][:4]
files=[cache / 'train' / r['episode_id'] / 'samples.jsonl' for r in rows]
torch.set_num_threads(2)
torch.manual_seed(cfg['seed'])
device=torch.device('cuda')
dataset=M3Dataset(files,history_dim=cfg['model']['history_dim'])
assert dataset.history_source == cfg['history_source']
verify_vision_source(cfg['model_path'],dataset.history_source)
processor=AutoProcessor.from_pretrained(cfg['model_path'],max_pixels=cfg['max_pixels'],local_files_only=True)
collate=SequenceCollator(processor,dataset.history_dim,cfg['max_sequence_length'],dataset.history_capacity)
windows=EpisodeWindows(dataset)
indices=[[i for i,(e,_,_) in enumerate(dataset.samples) if e==episode] for episode in range(4)]
started=time.monotonic()
model=M3ProgressPredictor(cfg['model_path'],**cfg['model']).to(device).train()
initialization=initialize_weights(model,cfg['init_checkpoint'],cfg['model'])
params=[p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=cfg.get('fused_optimizer',False))
assert len(optimizer.state)==0
torch.cuda.reset_peak_memory_stats()
checks=[]
state=None
def check_window(positions, carry, name):
    window={'indices':[[indices[b][positions[b]+t] for b in range(4)] for t in range(8)],'reset':carry is None,'group_end':True}
    batch=collate(windows[window])
    inputs=[to_device(frame,device) for frame in batch['inputs']]
    targets=batch['target_pt'].to(device)
    result=model(inputs,targets,state=carry)
    assert torch.isfinite(result['loss']) and torch.isfinite(result['pt']).all()
    assert result['pt'].shape==(32,) and result['num_samples'].item()==32
    result['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
    assert all(p.grad is None for p in model.qwen.model.visual.parameters())
    groups={
        'lora': any(p.grad is not None and p.grad.abs().sum().item()>0 for n,p in model.named_parameters() if 'lora_' in n),
        'history_attention': any(p.grad is not None and p.grad.abs().sum().item()>0 for module in model.memory_adapters.values() for p in module.parameters()),
        'progress_head': any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.progress_head.parameters()),
    }
    assert all(groups.values()),groups
    checks.append({'name':name,'observations':32,'loss':result['loss'].item(),'pt_min':result['pt'].min().item(),'pt_max':result['pt'].max().item(),'history_tokens_max':max(int(x['history_valid_mask'].sum(-1).max()) for x in inputs),'nonzero_gradients':groups,'recurrent_layers':len(result['state'].layers),'carried_previous_chunk_state':carry is not None})
    print(json.dumps(checks[-1]),flush=True)
    return result['state'].detach()
state=check_window([0]*4,None,'first_chunk')
state=check_window([8]*4,state,'continuation_chunk')
torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'],error_if_nonfinite=True)
optimizer.step()
assert len(optimizer.state)>0
assert all(torch.isfinite(p).all() for p in params)
optimizer.zero_grad(set_to_none=True)
seams=[]
for b,row in enumerate(rows):
    position=next(i for i,index in enumerate(indices[b]) if dataset.samples[index][1]['frame_index']>=row['second_round_start'])
    seams.append(max(0,position-4))
check_window(seams,None,'internal_round_boundary_with_full_forest_history')
optimizer.zero_grad(set_to_none=True)
report={'status':'passed','initialization':initialization,'batch_size':4,'sequence_window':8,'checks':checks,'optimizer_started_fresh':True,'optimizer_test_steps_discarded':1,'learning_rate':cfg['learning_rate'],'peak_cuda_memory_mib':torch.cuda.max_memory_allocated()/2**20,'seconds':time.monotonic()-started,'formal_training_started':False,'checkpoint_written':False,'training_sample_files':[str(p) for p in files]}
(run/'preflight-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
