from pathlib import Path
import json,time,torch
from robometer_training.model import load_processor,load_model,attach_lora,current_logits,progress_loss,progress_values
from robometer_training.data import ProgressCollator,ObservationWindow
from robometer_training.prepare_proprimem import build_episode
ROOT=Path('/cache/cvpzhr');RUN=ROOT/'setup/robometer-20261031';DATA=ROOT/'datasets/robometer-proprimem-5fps-w8-20261031'
torch.set_num_threads(2)
processor=load_processor(ROOT/'models/Robometer-4B',min_pixels=4096,max_pixels=65536)
model,settings=load_model(ROOT/'models/Robometer-4B',processor);attach_lora(model,rank=16,alpha=32,dropout=.05);model.cuda().eval()
source=ROOT/'datasets/forest-proprimem-5fps-20261030/cache/train/Count/01-press_buttons_by_number/episode_000900'
rows,meta=build_episode((str(source),str(DATA),'train'));selected=[rows[0],rows[7],rows[-1]]
collator=ProgressCollator(processor)
cached=collator.encode(selected);online=collator.encode([{k:v for k,v in r.items() if k not in ('pixel_cache','window_indices')} for r in selected])
assert cached.keys()==online.keys()
for k in cached:assert torch.equal(cached[k],online[k].to(cached[k].dtype)),k
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 a=current_logits(model,{k:v.cuda() for k,v in cached.items()});b=current_logits(model,{k:v.cuda() for k,v in online.items()})
assert torch.equal(a,b),(a-b).abs().max()
window=ObservationWindow(8)
for row in rows:
 actual=window.push(row['episode_id'],row['frame_index']);assert actual==row['history_frame_indices'];assert max(actual)<=row['frame_index']
assert window.push('next_round',100)==[100]*8
model.train();model.model.visual.eval();model.zero_grad(set_to_none=True)
with torch.autocast('cuda',dtype=torch.bfloat16):
 logits=current_logits(model,{k:v.cuda() for k,v in cached.items()});loss=progress_loss(logits,torch.tensor([r['pt'] for r in selected],device='cuda'))
loss.backward();trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
assert all('lora_' in n or 'progress_head.' in n for n,p in trainable)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in trainable)
assert any('lora_B' in n and bool(p.grad.abs().sum()>0) for n,p in trainable)
assert all(p.grad is None for p in model.model.visual.parameters())
report=dict(status='passed',strict_original_weight_load=True,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for n,p in trainable),cached_online_input_equal=True,cached_online_max_prediction_diff=float((progress_values(a)-progress_values(b)).abs().max()),loss=float(loss.detach()),all_trainable_gradients_finite=True,visual_frozen=True,causal_windows_and_round_reset=True,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,optimizer_steps=0,parameters_updated=False)
(RUN/'gpu-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
