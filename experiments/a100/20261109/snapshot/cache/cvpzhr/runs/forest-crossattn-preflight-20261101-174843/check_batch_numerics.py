import json
from pathlib import Path
import torch
from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.m3_inputs import M3Collator
from mem_lite.runtime import load_checkpoint,to_device
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

out=Path('/cache/cvpzhr/outputs/forest-crossattn-preflight-20261101-174843')
model,processor,config=load_checkpoint(out/'last.pt')
data=M3Dataset(['/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030/cache/validation/Count/01-press_buttons_by_number/episode_000904'])
rows=[data[0],data[15]]
collator=M3Collator(processor,data.history_dim,config['max_sequence_length'],data.history_capacity)
report={}
@torch.no_grad()
def compare(name,samples):
    mixed=model(to_device(collator(samples),'cuda'))['pt']
    singles=torch.cat([model(to_device(collator([r]),'cuda'))['pt'] for r in samples])
    report[name]={'batch':mixed.tolist(),'individual':singles.tolist(),'max_difference_pp':100*float((mixed-singles).abs().max())}
    print(name,report[name],flush=True)
with torch.no_grad():
    compare('bf16_trained_gates',rows)
    for module in model.memory_adapters.values():module.gate.zero_()
    compare('bf16_zero_gates',rows)
    plain=[{**r,'history_tokens':None} for r in rows]
    compare('bf16_no_history',plain)
    compare('bf16_same_sample_twice',[plain[0],plain[0]])
    # FLA only accepts low precision; use the official Torch fallback for this diagnostic.
    for module in model.modules():
        if hasattr(module,'chunk_gated_delta_rule'):
            module.chunk_gated_delta_rule=torch_chunk_gated_delta_rule
    compare('bf16_torch_delta_rule',plain)
    model.float()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    compare('fp32_no_history',plain)
(out/'batch-numerics.json').write_text(json.dumps(report,indent=2)+'\n')
