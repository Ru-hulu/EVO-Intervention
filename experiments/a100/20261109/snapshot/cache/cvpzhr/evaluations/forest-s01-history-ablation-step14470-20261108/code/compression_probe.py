"""Fixed observations: protected compression and explicit unmerged visual anchors."""
import json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from mem_lite.datasets.m3_dataset import M3Dataset,Stage3Collator
from mem_lite.runtime import load_checkpoint,to_device
from history_probe import ROOT,CHECKPOINT,dump

def main():
 torch.set_num_threads(4)
 ds=M3Dataset(['/cache/cvpzhr/datasets/forest-s01-decline-5fps-20261106/validation/episode_000055/samples.jsonl'])
 indices={s['frame_index']:i for i,(_,s,_) in enumerate(ds.samples)}
 audit=json.loads((ROOT/'memory-audit.json').read_text())
 assert len(audit['records'])==27
 assert all(r['exact_cache_match'] for r in audit['records'] if r['condition']=='original'), 'Rebuild mismatch prevents compression attribution'
 model,processor,cfg=load_checkpoint(str(CHECKPOINT))
 collate=Stage3Collator(processor,ds.history_dim,cfg['max_sequence_length'],ds.history_capacity)
 rows=[]
 with torch.inference_mode(),(ROOT/'compression-results.jsonl').open('w') as out:
  for frame in [520,628,680,728,800,900,940,960,1000]:
   item=ds[indices[frame]];base=to_device(collate([item])['inputs'],'cuda');n=len(item['history_tokens'])
   variants=[('original',base),('original_repeat',base)]
   def fixed(tokens):
    x=dict(base,history_tokens=torch.zeros_like(base['history_tokens']))
    x['history_tokens'][0,:n]=tokens.to(x['history_tokens'])
    assert all(torch.equal(base[k],x[k]) for k in base if k!='history_tokens')
    return x
   for name in ['protect_failure','protect_neutral']:
    tokens=torch.load(ROOT/'rebuilt'/f'{name}-{frame}.pt',weights_only=True)
    raw=to_device(collate([dict(item,history_tokens=tokens)])['inputs'],'cuda')
    assert torch.equal(raw['pixel_values'],base['pixel_values'])
    variants.append((name,raw))
    matched=F.adaptive_avg_pool1d(tokens.float().T[None],n)[0].T
    variants.append((name+'_same_length',fixed(matched)))
   if frame>=680:
    for name,frames in [('raw_failure_anchors',[0,600,628,660]),('raw_neutral_anchors',[0,200,228,260])]:
     tokens=torch.cat([torch.load(ROOT/'rebuilt'/f'raw-{f}.pt',weights_only=True) for f in frames])
     assert len(tokens)<n
     # Same token count; preserve original recent tail and absolute visual positions.
     replacement=torch.cat([tokens,item['history_tokens'][len(tokens):]])
     variants.append((name,fixed(replacement)))
    variants.append(('reverse_history_tokens',fixed(item['history_tokens'].flip(0))))
   for condition,inputs in variants:
    result=model(inputs);probs=result['bin_probs'][0].float().cpu()
    r=dict(current_frame=frame,condition=condition,predicted_pt=float(result['pt'][0]),p_zero=float(probs[0]),p_low=float(probs[:11].sum()),p_high=float(probs[80:].sum()),history_tokens=int(inputs['history_valid_mask'].sum()),target=item['target_pt'],probabilities=probs.tolist())
    rows.append(r);out.write(json.dumps(r)+'\n');out.flush()
   print('completed frame',frame,flush=True)
 dump(ROOT/'compression-status.json',dict(phase='completed',records=len(rows),baseline_cache_rebuilt_exactly=True,
      caution='Protection changes which background information is compressed. Fixed-length resampling, raw-anchor replacement and token reversal are diagnostic OOD inputs, not ground-truthed alternate rollouts.'))
if __name__=='__main__':main()
