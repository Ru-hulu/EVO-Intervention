"""Predeclared gates; never adjust tolerances in response to probe results."""
import json
from pathlib import Path
import statistics

LIMITS = dict(pt_abs=0.002,loss_abs=0.02,loss_relative=0.005,
              gradient_relative_l2=0.05,gradient_cosine=0.999,
              allocated_gib=16.0,reserved_gib=17.0,
              memory_relative=1.25,memory_extra_gib=1.0)


def compare(folder):
    import torch
    folder=Path(folder)
    original=torch.load(folder/'original.pt',map_location='cpu',weights_only=True)
    fast=torch.load(folder/'accelerated.pt',map_location='cpu',weights_only=True)
    assert original['initial_trainable_sha256']==fast['initial_trainable_sha256']
    assert original['config']==fast['config']
    assert len(original['cases'])==len(fast['cases'])==6
    records=[]
    for a,b in zip(original['cases'],fast['cases']):
        assert a['index']==b['index'] and a['sequence_length']==b['sequence_length']
        ga,gb=a['gradients'],b['gradients'];assert ga.keys()==gb.keys()
        dot=aa=bb=dd=0.0
        for name in ga:
            x,y=ga[name].float(),gb[name].float()
            assert torch.isfinite(x).all() and torch.isfinite(y).all()
            dot+=float((x*y).sum(dtype=torch.float64));aa+=float(x.square().sum(dtype=torch.float64))
            bb+=float(y.square().sum(dtype=torch.float64));dd+=float((x-y).square().sum(dtype=torch.float64))
        assert aa>0 and bb>0
        row=dict(index=a['index'],pt_abs=float((a['pt']-b['pt']).abs().max()),
                 loss_abs=float((a['loss']-b['loss']).abs()),original_loss=float(a['loss']),
                 gradient_relative_l2=(dd/aa)**.5,gradient_cosine=dot/(aa*bb)**.5,
                 original_seconds=a['seconds'],accelerated_seconds=b['seconds'])
        row['passed']=(row['pt_abs']<=LIMITS['pt_abs'] and
            row['loss_abs']<=LIMITS['loss_abs']+LIMITS['loss_relative']*abs(row['original_loss']) and
            row['gradient_relative_l2']<=LIMITS['gradient_relative_l2'] and
            row['gradient_cosine']>=LIMITS['gradient_cosine'])
        records.append(row)
    memory=fast['peak_allocated_gib']<=min(LIMITS['allocated_gib'],
        original['peak_allocated_gib']*LIMITS['memory_relative']+LIMITS['memory_extra_gib'])
    memory=memory and fast['peak_reserved_gib']<=LIMITS['reserved_gib']
    result=dict(status='passed' if all(r['passed'] for r in records) and memory else 'failed',
                numerical_passed=all(r['passed'] for r in records),memory_passed=memory,
                limits=LIMITS,cases=records,peak_allocated_gib={
                    'original':original['peak_allocated_gib'],'accelerated':fast['peak_allocated_gib']},
                warm_median_speedup=statistics.median(r['original_seconds'] for r in records[1:])/
                    statistics.median(r['accelerated_seconds'] for r in records[1:]))
    (folder/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(compare(sys.argv[1]),indent=2))
