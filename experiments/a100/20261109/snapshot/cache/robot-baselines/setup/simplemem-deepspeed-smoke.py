import json
from pathlib import Path
import torch
import torch.distributed as dist
import deepspeed

dist.init_process_group('nccl')
torch.cuda.set_device(0)
model = torch.nn.Linear(16, 4).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
config = {'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
          'zero_optimization': {'stage': 2}, 'zero_allow_untested_optimizer': True,
          'bf16': {'enabled': True}, 'steps_per_print': 100}
engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config, dist_init_required=False)
before = model.weight.detach().clone()
loss = engine(torch.ones(1, 16, device='cuda', dtype=torch.bfloat16)).float().square().mean()
engine.backward(loss)
engine.step()
assert torch.isfinite(loss)
assert not torch.equal(before, model.weight)
report = dict(zero_stage=2, bf16=True, loss=float(loss), optimizer_step=True)
Path('/cache/robot-baselines/manifests/simplemem-deepspeed-smoke.json').write_text(json.dumps(report, indent=2)+'\n')
print('DEEPSPEED_OPTIMIZER_STEP_OK', json.dumps(report), flush=True)
dist.destroy_process_group()
