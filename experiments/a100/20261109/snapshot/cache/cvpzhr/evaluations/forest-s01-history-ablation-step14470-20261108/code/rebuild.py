"""Rebuild memory with event provenance; protect event/neutral windows at same quota."""
import json,time
from pathlib import Path
import torch
from PIL import Image
from audit_manager import MemoryManager
from mem_lite.memory.vision import QwenFrameEncoder,load_vision_encoder
from history_probe import ROOT,dump
CACHE=Path('/cache/cvpzhr/datasets/forest-s01-decline-5fps-20261106/validation/episode_000055')
FRAMES=[520,628,680,728,800,900,940,960,1000]
def main():
 torch.set_num_threads(4)
 summary=json.loads((CACHE/'summary.json').read_text());origin=summary['history_source']
 visual,processor,digest=load_vision_encoder('/cache/cvpzhr/models/Qwen3.5-4B','cuda')
 assert digest==origin['vision_sha256']
 encoder=QwenFrameEncoder(visual)
 managers={key:MemoryManager(encoder.hidden_size,**origin['memory_config']) for key in ['original','protect_failure','protect_neutral']}
 managers['protect_failure'].protect=(125,165) # frames 500..660, onset through physical failure
 managers['protect_neutral'].protect=(50,90) # frames 200..360, matched 41-frame duration
 samples=[json.loads(x) for x in (CACHE/'samples.jsonl').read_text().splitlines()]
 audits=[];(ROOT/'rebuilt').mkdir(exist_ok=False)
 started=time.time()
 with torch.inference_mode():
  for row in samples:
   f=row['frame_index']
   if f>max(FRAMES):break
   if f in FRAMES:
    stored=torch.load(CACHE/row['history'],weights_only=True)['tokens']
    for name,m in managers.items():
     tokens=m.get_memory_tokens().cpu();assert len(tokens)<=summary['history_capacity']
     torch.save(tokens,ROOT/'rebuilt'/f'{name}-{f}.pt')
     audit=dict(condition=name,current_frame=f,through_frame=f-4,tokens=len(tokens),events=[dict(source_frames=[samples[i]['frame_index'] for i in e.sources],tokens=len(e.tokens),merges=e.merges) for e in m.memory_long])
     if name=='original':
      audit['shape_matches_cache']=list(tokens.shape)==list(stored.shape)
      audit['exact_cache_match']=torch.equal(tokens,stored)
      if tokens.shape==stored.shape:audit['max_abs_cache_diff']=float((tokens.float()-stored.float()).abs().max())
     audits.append(audit)
    dump(ROOT/'memory-audit.json',dict(records=audits,protected_frame_ranges={'failure':[500,660],'neutral':[200,360]},long_token_quota=800,seconds=time.time()-started))
   if f==max(FRAMES):break
   with Image.open(CACHE/row['images']['cam_high']) as im:
    inputs=processor(images=[im.convert('RGB')],return_tensors='pt',max_pixels=origin['max_pixels'],min_pixels=origin['min_pixels'])
   tokens=encoder(inputs['pixel_values'],inputs['image_grid_thw'],[f])[0]
   if f in [0,200,228,260,500,560,600,620,628,644,660]:
    torch.save(tokens.cpu(),ROOT/'rebuilt'/f'raw-{f}.pt')
   for m in managers.values():m.update(tokens)
   if f%200==0:print(json.dumps(dict(frame=f,seconds=time.time()-started)),flush=True)
 print('Reconstruction completed',flush=True)
if __name__=='__main__':main()
