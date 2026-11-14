from copy import deepcopy
import pytest
from experiments.prm_five_groups_20261114.group1_report import summarize,verify_record
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics


def test_partial_suite_does_not_publish_selection_biased_mean():
    row=trajectory_metrics(list(range(8)),list(range(8)),list(range(8)))
    partial=summarize([row],2,16)
    assert not partial['coverage_complete'] and partial['defined_trajectory_means']['pearson'] is None
    complete=summarize([row,row],2,16)
    assert complete['coverage_complete'] and complete['defined_trajectory_means']['pearson']==1
    with pytest.raises(AssertionError):summarize([row,row],2,17)


def test_undefined_and_invalid_denominators_are_explicit():
    good=trajectory_metrics(list(range(8)),list(range(8)),list(range(8)))
    flat=trajectory_metrics([1]*8,list(range(8)),list(range(8)))
    bad=trajectory_metrics([None]*8,list(range(8)),list(range(8)))
    r=summarize([good,flat,bad],3,24)
    assert r['coverage_complete'] and r['invalid_frames']==8 and r['invalid_episodes']==1
    assert r['defined_episode_counts']['pearson']==1 and r['undefined_episode_counts']['pearson']==2
    assert r['correlation_undefined_reasons']=={'constant_prediction':1,'invalid_predictions':1}


def test_record_truth_frames_raw_and_future_reference_are_checked():
    e=dict(key='Count/C01/e',suite='Count',task_id='C01',frame_indices=list(range(8)),
           target_pt=list(range(8)),timestamps=list(range(8)),progress_sha256='pt',video_hashes=['v',None,None])
    r=dict(e,model='test',group=1,condition='baseline',input_fps=5,manifest_sha256='manifest',plan_sha256='plan',
           source_video_sha256=e['video_hashes'],predictions=list(range(8)),raw_responses=[None]*8,uses_future_endpoint_reference=False)
    assert verify_record(r,e,'test','manifest','plan',False)['pearson']==1
    for field,value in [('target_pt',[0]*8),('frame_indices',list(range(1,9))),('raw_responses',[]),('uses_future_endpoint_reference',True),('plan_sha256','changed')]:
        broken=deepcopy(r);broken[field]=value
        with pytest.raises(AssertionError):verify_record(broken,e,'test','manifest','plan',False)
