"""Prepare one shared augmented validation timeline on CPU; preserve all training inputs."""
from collections import Counter
import json, os, sys
from common import *
import prepare_static_jitter as aug
def main():
    assert not (ROOT/'plan.json').exists(), 'Plan already exists; use the immutable existing plan'
    OUT.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    source_checkpoint=Path('/cache/cvpzhr/outputs/forest-original-cap50-static-jitter-e3-20261104/last.pt')
    if not F_CKPT.exists():os.link(source_checkpoint,F_CKPT)
    import torch
    c=torch.load(F_CKPT,map_location='cpu',weights_only=True,mmap=True)
    assert (c['epoch'],c['next_batch'],c['step'])==(1,0,19934)
    del c
    validation=read(BASE/'validation.sample_files.json')
    training=read(BASE/'train.sample_files.json')
    training_ids={read(Path(x)/'summary.json')['episode_id'] for x in training}
    assert len(validation)==417
    contract=dict(aug.CONTRACT)
    contract.update(scope='417 validation episodes, all 47 tasks; one deterministic augmented version each',
                    validation='independent augmented evaluation; training and clean validation unchanged')
    episodes=[];all_counts=Counter();common=None;all_suite=defaultdict(Counter)
    for directory in validation:
        source=Path(directory);summary=read(source/'summary.json');rr=rows(source/'samples.jsonl')
        eid=summary['episode_id'];suite=summary['source']['suite'];task=summary['source']['task']
        assert eid not in training_ids and eid!='Order/03-swap_blocks_with_confirmation/episode_001342'
        assert summary['source']['split']=='validation'
        if common is None:common=summary['history_source']
        assert common==summary['history_source']
        fps=aug.source_fps(summary)
        events=aug.choose_events(rr,eid,fps,summary['source']['frames'])
        expanded=list(aug.expand(rr,events,fps))
        original=rows(RB_CACHE/'episodes/validation'/(eid+'.jsonl'))
        assert len(rr)==len(original)
        for j,(a,b) in enumerate(zip(rr,original,strict=True)):
            assert a['pt']==b['pt'] and a['frame_index']==b['source_frame_index'] and b['sample_index']==j
        target=ROOT/'timeline'/(eid+'.jsonl');target.parent.mkdir(parents=True,exist_ok=True)
        rtarget=ROOT/'robometer'/(eid+'.jsonl');rtarget.parent.mkdir(parents=True,exist_ok=True)
        with target.open('w') as f,rtarget.open('w') as rf:
            for i,row in enumerate(expanded):
                source_i=row['source_sample_index'];anchor=row['anchor_sample_index']
                assert row['pt']==rr[anchor]['pt'] and row['images']==rr[source_i]['images']
                if i:assert row['frame_index']>expanded[i-1]['frame_index']
                if row['kind']!='original':assert row['pt']==expanded[i-1]['pt']
                meta=dict(row,episode_id=eid,suite=suite,task=task)
                f.write(json.dumps(meta,ensure_ascii=False)+'\n')
                past=[max(0,j) for j in range(i-7,i)]
                indices=past+[i,i,i]
                source_indices=[expanded[j]['source_sample_index'] for j in indices]
                views=['cam_high']*8+['cam_left_wrist','cam_right_wrist']
                image=[original[j]['image'][slot] for j,slot in zip(source_indices,[-3]*8+[-2,-1],strict=True)]
                rec=dict(original[source_i],id=f'{eid}/{i:08d}',
                    **{k:row[k] for k in ('sample_index','frame_index','source_frame_index','source_sample_index',
                                          'anchor_sample_index','kind','timestamp_seconds','pt')})
                rec.update(image=image,input_views=views,history_sample_indices=past,
                    observation_indices=indices,window_indices=source_indices,
                    history_frame_indices=[expanded[j]['frame_index'] for j in past],
                    history_source_frame_indices=[expanded[j]['source_frame_index'] for j in past])
                assert rec['image'][-3:]==original[source_i]['image'][-3:]
                rf.write(json.dumps(rec,ensure_ascii=False)+'\n')
        counts=Counter(r['kind'] for r in expanded);all_counts.update(counts)
        all_suite[suite].update(episodes=1,original_samples=len(rr),augmented_samples=len(expanded))
        episodes.append(dict(episode_id=eid,task=task,suite=suite,source_dir=str(source),
             output_dir=str(CACHE/'cache/validation'/eid),
             source_samples_sha256=sha(source/'samples.jsonl'),source_summary_sha256=sha(source/'summary.json'),
             source_samples=len(rr),augmented_samples=len(expanded),added_samples=len(expanded)-len(rr),
             events=events,timeline=str(target),timeline_sha256=sha(target),
             robometer_rows=str(rtarget),robometer_rows_sha256=sha(rtarget),kinds=dict(counts),
             pixel_cache=original[0]['pixel_cache']))
    assert sum(e['source_samples'] for e in episodes)==66891
    assert len({e['episode_id'] for e in episodes})==417 and len({e['task'] for e in episodes})==47
    loads=[0]*4
    for e in sorted(episodes,key=lambda e:(-e['augmented_samples'],e['episode_id'])):
        rank=min(range(4),key=lambda r:loads[r]);e['worker']=rank;loads[rank]+=e['augmented_samples']
    protected=[BASE/'train.sample_files.json',BASE/'validation.sample_files.json',
               R_SETUP/'config.json',R_SETUP/'training-inputs.json']
    protected+=list(F_CODE.rglob('*.py'))+list((R_CODE/'robometer_training').glob('*.py'))
    protected += [ROOT/'prepare_static_jitter.py',ROOT/'prepare_repeats.py',ROOT/'augmentation-policy.json']
    p=dict(created_at=now(),contract=contract,episodes=episodes,model='/cache/cvpzhr/models/Qwen3.5-4B',
           history_source=common,code=str(F_CODE),forest_checkpoint=str(F_CKPT),
           forest_checkpoint_sha256=sha(F_CKPT),forest_epoch=1,forest_step=19934,
           robometer_checkpoint=str(R_CKPT),robometer_epoch=1,robometer_step=9967,
           checkpoint_selection='final completed epoch 1 for both models; no selection on augmented validation',
           cache_output=str(CACHE),evaluation_output=str(OUT),workers=4,worker_samples=loads,
           original_samples=66891,augmented_samples=sum(e['augmented_samples'] for e in episodes),
           kinds=dict(all_counts),suites=dict(all_suite),
           metric='Pearson(predicted_pt, target_pt) within each episode, arithmetic mean across episodes',
           undefined_policy='null per episode, disclose count and IDs; do not silently impute',
           protected_hashes={str(p):sha(p) for p in protected})
    p['plan_sha256']=aug.canonical_hash(p)
    write(ROOT/'plan.json',p);write(OUT/'plan.json',p)
    write(ROOT/'preparation.json',dict(phase='completed',episodes=417,tasks=47,suites=p['suites'],
          samples=p['augmented_samples'],kinds=p['kinds'],gpu_used=False,plan_sha256=p['plan_sha256'],
          robometer_pixels='Reuse unchanged validation image tensors by source observation index; history follows augmented timeline'))
    print(json.dumps(read(ROOT/'preparation.json'),ensure_ascii=False))
if __name__=='__main__':main()
