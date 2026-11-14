from copy import deepcopy
import pytest
from experiments.prm_five_groups_20261114.stream_manifests import (
    disturbance_events,playback_map,fps_map,score_fps_stream,freeze)


def row(n=200):
    indices=list(range(0,n*6,6))
    return {'key':'State/S01/trajectory','suite':'State','task_id':'S01','frames':n*6,
        'source_rate':'30','frame_indices':indices,'timestamps':[i/30 for i in indices],
        'target_pt':[i/(n-1) for i in range(n)],'progress_sha256':'frozen-source-label',
        'input_issues':[{'view':1,'reason':'short_wrist'}]}


def test_static_map_preserves_every_original_frame_and_flat_truth():
    r=row();events=[{'after_source_step':20,'duration_steps':5},{'after_source_step':60,'duration_steps':15}]
    m=playback_map(r,'static_only',42,events);assert len(m['target_pt'])==220
    removed=set()
    for event in m['events']:
        start,end=event['first_inserted_step'],event['last_inserted_step']+1
        removed.update(range(start,end));anchor=event['after_source_step']
        assert m['target_pt'][start:end]==[r['target_pt'][anchor]]*(end-start)
        assert m['source_frame_indices'][start:end]==[r['frame_indices'][anchor]]*(end-start)
    assert [x for i,x in enumerate(m['source_frame_indices']) if i not in removed]==r['frame_indices']
    assert m['playback_timestamps'][-1]==pytest.approx(r['timestamps'][-1]+4)
    assert m['input_issues']==r['input_issues'] and m['all_views_share_source_frame_map']


def test_rewind_reverses_then_returns_without_resetting_memory_or_truth():
    r=row();events=[{'after_source_step':20,'duration_steps':5},{'after_source_step':60,'duration_steps':15}]
    m=playback_map(r,'rewind_only',42,events);assert len(m['target_pt'])==240
    assert m['source_step_indices'][21:31]==[19,18,17,16,15,16,17,18,19,20]
    assert m['target_pt'][21:31]==[r['target_pt'][i] for i in [19,18,17,16,15,16,17,18,19,20]]
    assert m['source_step_indices'][31]==21
    assert all(a<b for a,b in zip(m['playback_timestamps'],m['playback_timestamps'][1:]))
    assert m['playback_timestamps'][-1]==pytest.approx(r['timestamps'][-1]+8)
    assert m['memory_enabled'] and m['memory_advances_in']=='presented playback order'


def test_maps_are_order_independent_and_seeded_without_mutating_inputs():
    r=row();original=deepcopy(r)
    first=playback_map(r,'rewind_only',42)
    playback_map(dict(r,key='another'), 'rewind_only',43)
    assert first==playback_map(r,'rewind_only',42) and r==original
    assert first['map_sha256']!=playback_map(r,'rewind_only',43)['map_sha256']
    e=disturbance_events(r['key'],200,'static_only',44)
    assert len({x['after_source_step'] for x in e})==2
    assert all(5<=x['duration_steps']<=15 and 15<=x['after_source_step']<199 for x in e)
    with pytest.raises(ValueError):disturbance_events('short',17,'rewind_only',42)
    plain=playback_map(r,'no_memory')
    assert plain['source_frame_indices']==r['frame_indices'] and plain['target_pt']==r['target_pt']
    assert not plain['memory_enabled'] and not plain['events']


def test_all_fps_share_exact_original_truth_and_integer_second_grid():
    r=row();r['frames']=301;truth={i:(i/300)**2 for i in range(301)}
    entries=[fps_map(r,truth,f) for f in [30,20,10,5,2]]
    for e in entries:
        assert e['target_pt']==[truth[i] for i in e['source_frame_indices']]
        assert e['source_frame_indices'][-1]==300
        assert e['common_scoring_native_frames']==list(range(0,301,30))
        assert [e['source_frame_indices'][i] for i in e['common_scoring_indices']]==list(range(0,301,30))
        assert e['common_scoring_timestamps']==list(range(11))
    assert entries[1]['source_frame_indices'][:5]==[0,2,3,5,6]
    with pytest.raises(ValueError):fps_map(dict(r,source_rate='25'),truth,30)
    with pytest.raises(ValueError):fps_map(r,truth,15)


def test_common_grid_scoring_keeps_full_stream_invalid_accounting():
    r=row();r['frames']=301;truth={i:i/300 for i in range(301)}
    e=fps_map(r,truth,20);pred=list(e['target_pt']);pred[1]=None
    scores=score_fps_stream(pred,e)
    assert scores['full_stream']['pearson'] is None
    assert scores['full_stream']['invalid_predictions']==1
    assert scores['common_integer_seconds']['pearson']==pytest.approx(1)
    assert scores['common_integer_seconds']['scoring_fps']==1
    assert not scores['common_integer_seconds']['interpolated']
    pred[0]=None
    assert score_fps_stream(pred,e)['common_integer_seconds']['pearson'] is None


def test_frozen_manifest_refuses_overwrite(tmp_path):
    p=tmp_path/'map.json';a={'target_pt':[0,.2,1]}
    assert freeze(p,a)==freeze(p,a)
    with pytest.raises(ValueError):freeze(p,{'target_pt':[0,.5,1]})
