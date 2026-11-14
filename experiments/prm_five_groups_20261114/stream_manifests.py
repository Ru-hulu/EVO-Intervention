"""Frozen playback maps for group 2 and matched native-30-FPS group 5 inputs.

This prepares inputs and scoring maps only; it never launches inference. Every
view uses the same source-frame map, and memory advances in playback order.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import random

from .eval_common import EVAL, PLAN, sampling_indices, trajectory_metrics
from .prepare_cache import progress_values, save, sha


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError(f'Frozen input changed: {path}')
    else:
        save(path,value)
    return sha(path)


def disturbance_events(key,length,variant,seed):
    if variant not in ('static_only','rewind_only') or seed not in (42,43,44):
        raise ValueError('Unsupported confirmed disturbance condition')
    # A fixed full-duration-capable interior prevents rejection sampling from
    # shortening the requested uniform 5..15-step duration distribution.
    if length<18:
        raise ValueError('Too short for two distinct interior points with a 3-second prefix')
    rng=random.Random(int(digest({'key':key,'variant':variant,'seed':seed}),16))
    anchors=rng.sample(range(15,length-1),2)
    return sorted([{'after_source_step':a,'duration_steps':rng.randint(5,15)} for a in anchors],
                  key=lambda e:e['after_source_step'])


def playback_map(row,variant,seed=None,events=None):
    indices=row['frame_indices'];truth=row['target_pt'];times=row['timestamps'];n=len(indices)
    assert len(truth)==len(times)==n and all(a<b for a,b in zip(times,times[1:]))
    if variant=='no_memory':
        assert seed is None and events is None
        events=[]
    elif events is None:
        events=disturbance_events(row['key'],n,variant,seed)
    if variant not in ('no_memory','static_only','rewind_only'):
        raise ValueError('Unsupported playback variant')
    if variant!='no_memory':
        assert len(events)==2
    event_by_anchor={e['after_source_step']:e for e in events}
    assert len(event_by_anchor)==len(events)
    source_steps=[];playback_times=[];audits=[];inserted=0
    for i in range(n):
        source_steps.append(i);playback_times.append(times[i]+inserted/5)
        if i not in event_by_anchor:continue
        event=event_by_anchor[i];length=event['duration_steps']
        assert 5<=length<=15 and 0<i<n-1
        if variant=='rewind_only':
            assert i>=length
            extra=list(range(i-1,i-length-1,-1))+list(range(i-length+1,i+1))
        else:
            extra=[i]*length
        start=len(source_steps);anchor_time=playback_times[-1]
        source_steps.extend(extra)
        playback_times.extend(anchor_time+j/5 for j in range(1,len(extra)+1))
        inserted+=len(extra)
        audits.append(dict(event,first_inserted_step=start,last_inserted_step=len(source_steps)-1,
                           inserted_steps=len(extra),one_way_duration_seconds=length/5))
    assert len(audits)==len(events) and all(a<b for a,b in zip(playback_times,playback_times[1:]))
    entry={'key':row['key'],'suite':row['suite'],'task_id':row['task_id'],
        'condition':variant,'seed':seed,'input_fps':5,'scoring_fps':5,
        'source_step_indices':source_steps,'source_frame_indices':[indices[i] for i in source_steps],
        'source_timestamps':[times[i] for i in source_steps],'playback_timestamps':playback_times,
        'target_pt':[truth[i] for i in source_steps],'events':audits,
        'input_issues':row.get('input_issues',[]),'progress_sha256':row['progress_sha256'],
        'memory_enabled':variant!='no_memory','memory_advances_in':'presented playback order',
        'all_views_share_source_frame_map':True}
    entry['map_sha256']=digest(entry)
    return entry


def fps_map(row,native_truth,target_fps):
    if Fraction(row['source_rate'])!=30 or target_fps not in (30,20,10,5,2):
        raise ValueError('Group 5 is restricted to native 30 FPS and the five confirmed rates')
    assert list(native_truth)==list(range(row['frames']))
    indices=sampling_indices(row['frames'],row['source_rate'],target_fps)
    source_to_step={f:i for i,f in enumerate(indices)}
    common_native=list(range(0,row['frames'],30))
    assert all(f in source_to_step for f in common_native)
    entry={'key':row['key'],'suite':row['suite'],'task_id':row['task_id'],
        'condition':f'fps{target_fps}','input_fps':target_fps,'source_rate':'30',
        'source_frame_indices':indices,'playback_timestamps':[i/30 for i in indices],
        'target_pt':[native_truth[i] for i in indices],
        'common_scoring_indices':[source_to_step[f] for f in common_native],
        'common_scoring_native_frames':common_native,'common_scoring_fps':1,
        'common_scoring_timestamps':[f/30 for f in common_native],
        'input_issues':row.get('input_issues',[]),'progress_sha256':row['progress_sha256'],
        'memory_enabled':True,'memory_configuration':'unchanged native configuration',
        'interpolate_predictions':False,'all_views_share_source_frame_map':True}
    entry['map_sha256']=digest(entry)
    return entry


def score_fps_stream(predictions,entry):
    assert len(predictions)==len(entry['source_frame_indices'])
    common=entry['common_scoring_indices'];truth=entry['target_pt'];times=entry['playback_timestamps']
    return {'full_stream':{'scoring_fps':entry['input_fps'],
                          **trajectory_metrics(predictions,truth,times)},
        'common_integer_seconds':{'scoring_fps':1,'interpolated':False,
            **trajectory_metrics([predictions[i] for i in common],[truth[i] for i in common],
                                 entry['common_scoring_timestamps'])},
        'input_fps':entry['input_fps'],'map_sha256':entry['map_sha256']}


def build():
    baseline=EVAL/'group1-manifest.json';manifest=json.loads(baseline.read_text())
    assert manifest['plan_sha256']==sha(PLAN)
    rows=manifest['episodes'];assert len(rows)==385
    plan=json.loads(PLAN.read_text());groups={g['id']:g for g in plan['groups']}
    assert groups[2]['disturbances']['seeds']==[42,43,44]
    assert groups[2]['disturbances']['insertions_per_trajectory_per_variant']==2
    root=EVAL/'streams';receipts={}
    def persist(group,condition,entries,**extra):
        value={'schema':'prm-playback-maps-v1','group':group,'condition':condition,
            'plan_sha256':sha(PLAN),'baseline_manifest_sha256':sha(baseline),
            'code_sha256':sha(__file__),'expected_episodes':len(entries),
            'expected_samples':sum(len(e['source_frame_indices']) for e in entries),
            'by_suite':dict(Counter(e['suite'] for e in entries)),
            'source_media':'resolve original video paths from the frozen baseline manifest',
            'episodes':entries,**extra}
        path=root/f'group{group}'/f'{condition}.json'
        receipts[str(path)]={'sha256':freeze(path,value),'episodes':len(entries),
                            'samples':value['expected_samples'],'by_suite':value['by_suite']}
    persist(2,'no_memory',[playback_map(r,'no_memory') for r in rows])
    for variant in ('static_only','rewind_only'):
        for seed in (42,43,44):
            persist(2,f'{variant}-seed{seed}',[playback_map(r,variant,seed) for r in rows],seed=seed)
    subset=[r for r in rows if Fraction(r['source_rate'])==30]
    assert Counter(r['suite'] for r in subset)=={'State':59,'Order':109,'Count':67}
    native={r['key']:progress_values(r) for r in subset}
    for fps in (30,20,10,5,2):
        persist(5,f'fps{fps}',[fps_map(r,native[r['key']],fps) for r in subset],input_fps=fps,
                common_scoring_fps=1,extra_model_calls_for_scoring=False)
    receipt={'status':'verified_prepared','inference_launched':False,
        'group2_original_episodes':385,'group2_no_memory_versions':385,
        'group2_disturbed_versions':385*2*3,'group5_matched_native30_episodes':235,
        'group5_fps_versions':235*5,'manifests':receipts,
        'frame_map_policy':'native source PT lookup; no relabeling, smoothing, interpolation or new media',
        'runner_integration':'pending; this receipt does not mean group 2/5 inference is ready'}
    freeze(root/'preparation.json',receipt)
    return {k:v for k,v in receipt.items() if k!='manifests'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['build']);parser.parse_args()
    print(json.dumps(build(),ensure_ascii=False))
