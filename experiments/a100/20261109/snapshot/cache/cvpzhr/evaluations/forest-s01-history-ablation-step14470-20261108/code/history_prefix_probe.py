"""Within-episode history-age control for the S01 frame-900 failure."""
import json
import torch
from torch.nn import functional as F
from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.runtime import load_checkpoint, to_device
from history_probe import ROOT, CHECKPOINT, dump


def main():
    torch.set_num_threads(4)
    ds=M3Dataset(['/cache/cvpzhr/datasets/forest-s01-decline-5fps-20261106/validation/episode_000055/samples.jsonl'])
    indices={s['frame_index']:i for i,(_,s,_) in enumerate(ds.samples)}
    current=ds[indices[900]]
    model,processor,cfg=load_checkpoint(str(CHECKPOINT))
    collate=Stage3Collator(processor,ds.history_dim,cfg['max_sequence_length'],ds.history_capacity)
    base=to_device(collate([current])['inputs'],'cuda')
    results=[]
    with torch.inference_mode():
        for frame in [520,628,680,728,800,900]:
            donor=ds[indices[frame]]
            raw=to_device(collate([dict(current,history_tokens=donor['history_tokens'])])['inputs'],'cuda')
            fixed=dict(base,history_tokens=torch.zeros_like(base['history_tokens']))
            n=len(current['history_tokens'])
            fixed['history_tokens'][0,:n]=F.adaptive_avg_pool1d(donor['history_tokens'].to('cuda').float().T[None],n)[0].T.to(fixed['history_tokens'].dtype)
            assert torch.equal(base['pixel_values'],raw['pixel_values'])
            assert all(torch.equal(base[k],fixed[k]) for k in base if k!='history_tokens')
            for condition,inputs in [('raw_prefix',raw),('same_length_prefix',fixed)]:
                result=model(inputs);probs=result['bin_probs'][0].float().cpu()
                results.append(dict(current_episode=55,current_frame=900,history_sample_frame=frame,
                    history_through_frame=ds.samples[indices[frame]][2],history_tokens=len(donor['history_tokens']),
                    condition=condition,predicted_pt=float(result['pt'][0]),probabilities=probs.tolist(),
                    p_zero=float(probs[0]),p_low=float(probs[:11].sum()),p_high=float(probs[80:].sum())))
    dump(ROOT/'prefix-results.json',dict(records=results,note='Same current frame 900 and scene. Earlier cached histories omit the intervening observations; this is an input intervention, not a normal causal rollout.'))
    print(json.dumps([{k:v for k,v in r.items() if k!='probabilities'} for r in results]),flush=True)


if __name__=='__main__':main()
