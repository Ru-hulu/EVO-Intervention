"""CPU equivalence checks: cached pixels vs the exact online augmented ten-image input."""
import sys
from common import *
sys.path.insert(0,str(R_CODE))
sys.path.insert(0,str(R_CODE/'third_party/robometer-352d160389daa964788de1ec933d1925f3a6de4f'))
import torch
from robometer_training.data import ProgressCollator,_episode_cache,ObservationWindow
from robometer_training.model import load_processor
def main():
    torch.set_num_threads(2)
    p=plan();processor=load_processor('/cache/cvpzhr/models/Robometer-4B',min_pixels=4096,max_pixels=65536)
    collator=ProgressCollator(processor,4096)
    selected={}
    # One episode per task plus every distinct missing-view pattern.
    for e in p['episodes']:
        rr=rows(e['robometer_rows'])
        selected.setdefault(e['task'],e)
        selected.setdefault('missing:'+str(rr[0]['missing_current_views']),e)
    checked=0;kinds=set();mixed=[]
    for e in {v['episode_id']:v for v in selected.values()}.values():
        rr=rows(e['robometer_rows']);window=ObservationWindow(8)
        for r in rr:
            current=r['image'][-3:]
            self_window=window.push(r['episode_id'],current[0])
            assert self_window[:-1]+current==r['image']
        chosen={0,len(rr)//2,len(rr)-1}
        for kind in ('stationary','reverse','forward'):
            index=next((i for i,r in enumerate(rr) if r['kind']==kind),None)
            if index is not None:chosen.update(j for j in (index-1,index,index+1) if 0<=j<len(rr))
        batch=[rr[i] for i in sorted(chosen)]
        for start in range(0,len(batch),4):
            sub=batch[start:start+4]
            a=collator.encode(sub)
            b=collator.encode([{k:v for k,v in r.items() if k!='pixel_cache'} for r in sub])
            for key in a:
                aa,bb=a[key],b[key]
                if key=='pixel_values':aa,bb=aa.to(torch.bfloat16),bb.to(torch.bfloat16)
                assert torch.equal(aa,bb),(e['episode_id'],key)
            assert a['image_grid_thw'].shape[0]==10*len(sub)
            token=processor.tokenizer.convert_tokens_to_ids('<|prog_token|>')
            assert torch.all((a['input_ids']==token).sum(1)==8)
            checked+=len(sub);kinds.update(r['kind'] for r in sub)
        mixed.append(rr[len(rr)//2]);_episode_cache.cache_clear()
    for start in range(0,len(mixed),8):
        sub=mixed[start:start+8]
        a=collator.encode(sub);b=collator.encode([{k:v for k,v in r.items() if k!='pixel_cache'} for r in sub])
        for key in a:
            aa,bb=a[key],b[key]
            if key=='pixel_values':aa,bb=aa.to(torch.bfloat16),bb.to(torch.bfloat16)
            assert torch.equal(aa,bb),('mixed_tasks',key)
        _episode_cache.cache_clear()
    assert kinds=={'original','stationary','reverse','forward'}
    write(ROOT/'input-checks.json',dict(status='passed',episodes=len(mixed),tasks=47,checked_windows=checked,
          mixed_windows=len(mixed),kinds=sorted(kinds),cached_online_equal=True,online_history_equal=True,
          images_per_observation=10,progress_tokens=8,gpu_used=False,completed_at=now()))
    print(json.dumps(read(ROOT/'input-checks.json')))
if __name__=='__main__':main()
