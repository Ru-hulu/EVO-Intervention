"""Compare the fixed problematic validation trajectory before/after v4 training."""
import json
from pathlib import Path

BASE=Path('/cache/cvpzhr/evaluations')
OLD=BASE/'forest-s01-critical-val12-step13224-20261106'
NEW=BASE/'forest-s01-history-pairs-val12-step14604-20261107'
SOURCE=Path('/cache/robot-baselines/data/ProPriMem-S01-IMPRINT-Mixed-20261106-v4-history-pairs/State/01-cover_uncover_blocks')


def rows(root):
    return [json.loads(s) for s in (root/'predictions.jsonl').read_text().splitlines()
            if json.loads(s)['episode_index']==55]


def above(records,threshold,end_frame):
    total=longest=run=0.
    spans=[];start=None
    for i,r in enumerate(records):
        next_frame=records[i+1]['frame_index'] if i+1<len(records) else end_frame
        duration=(next_frame-r['frame_index'])/20
        if r['predicted_pt']>threshold:
            if start is None:start=r['frame_index']
            run+=duration;total+=duration;longest=max(longest,run)
        else:
            if start is not None:spans.append([start/20,r['frame_index']/20])
            run=0.;start=None
    if start is not None:spans.append([start/20,end_frame/20])
    return dict(total_seconds=total,longest_seconds=longest,intervals_seconds=spans)


def summarize(records,fail,n):
    post=[r for r in records if r['frame_index']>=fail]
    old_region=[r for r in records if 724<=r['frame_index']<=952]
    assert all(r['target_pt']==0 for r in post)
    peak=max(post,key=lambda r:r['predicted_pt'])
    return dict(full_mae_pp=sum(abs(r['predicted_pt']-r['target_pt']) for r in records)/len(records)*100,
        post_failure_mae_pp=sum(r['predicted_pt'] for r in post)/len(post)*100,
        post_failure_peak_pp=peak['predicted_pt']*100,peak_frame=peak['frame_index'],
        above20=above(post,.2,n),above80=above(post,.8,n),
        old_error_region_mae_pp=sum(r['predicted_pt'] for r in old_region)/len(old_region)*100,
        probes=[dict(frame=r['frame_index'],time=r['frame_index']/20,prediction_pp=r['predicted_pt']*100,
                     bin0_probability=r['bin_probs'][0],top_bin=max(range(101),key=r['bin_probs'].__getitem__))
                for r in records if r['frame_index'] in [628,680,720,732,800,900,940,952,960,1000]])


def main():
    old,new=rows(OLD),rows(NEW)
    assert [(r['frame_index'],r['target_pt']) for r in old]==[(r['frame_index'],r['target_pt']) for r in new]
    states=[json.loads(s) for s in (SOURCE/'evidence/episode_000055/states.jsonl').read_text().splitlines()]
    fail=next(i for i,s in enumerate(states) if s['fail_flag'])
    assert fail==626
    result=dict(episode=55,native_failure_frame=fail,native_failure_time=fail/20,
        comparison='same validation episode, same 5 Hz observations, causal histories and GT; different trained checkpoints',
        duration_convention='5 Hz prediction held to next observation; last held to video end',
        old_step13224=summarize(old,fail,len(states)),new_step14604=summarize(new,fail,len(states)))
    (NEW/'episode55-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
