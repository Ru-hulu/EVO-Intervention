from copy import deepcopy
import json
import pytest
from experiments.prm_five_groups_20261114.group1_report import summarize,verify_record
from experiments.prm_five_groups_20261114.eval_common import trajectory_metrics
from experiments.prm_five_groups_20261114.group1_report import common_comparison,COMPARISON_POLICY


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


def comparison_fixture():
    policy=json.loads(COMPARISON_POLICY.read_text());excluded=policy['excluded_episodes'][0]['key']
    entries=[dict(key='State/S01/a',suite='State',target_pt=list(range(8))),
             dict(key='Count/C01/a',suite='Count',target_pt=list(range(8))),
             dict(key='Order/O01/a',suite='Order',target_pt=list(range(8))),
             dict(key='Order/O02/a',suite='Order',target_pt=list(range(8))),
             dict(key=excluded,suite='Order',target_pt=list(range(88)),
                  input_issues=[dict(view=1,reason='view_frame_count_mismatch')])]
    rows=[]
    for entry in entries:
        truth=entry['target_pt'];pred=(list(reversed(truth)) if entry['key']==excluded else
            [1]*8 if entry['key']=='Order/O02/a' else truth)
        rows.append(dict(key=entry['key'],suite=entry['suite'],**trajectory_metrics(pred,truth,list(range(len(truth))))))
    def model(items):return dict(regime='official_zero_shot',uses_future_endpoint_reference=False,by_episode=items)
    report=dict(updated=1,manifest_sha256='manifest',plan_sha256='plan',aggregation='equal trajectories',
                models={'three_view':model(rows[:-1]),'main_only':model(rows)})
    return report,dict(episodes=entries,expected_episodes=5),policy


def test_common_order_exclusion_applies_even_when_model_predicted_missing_view_episode():
    report,manifest,policy=comparison_fixture();unchanged=deepcopy(report)
    result=common_comparison(report,manifest,policy)
    assert report==unchanged and result['expected_episodes']==4 and result['expected_samples']==32
    a,b=result['models']['three_view'],result['models']['main_only']
    assert a['by_suite']==b['by_suite'] and a['overall']==b['overall']
    order=b['by_suite']['Order']
    assert order['coverage_complete'] and order['defined_trajectory_means']['pearson']==1
    assert order['defined_episode_counts']['pearson']==1 and order['undefined_episode_counts']['pearson']==1
    assert len(b['excluded_result_was_present'])==1 and not a['excluded_result_was_present']


def test_comparison_exclusion_cannot_hide_other_missing_trajectories_or_be_expanded():
    report,manifest,policy=comparison_fixture()
    report['models']['main_only']['by_episode']=[r for r in report['models']['main_only']['by_episode'] if r['key']!='Order/O01/a']
    result=common_comparison(report,manifest,policy)
    order=result['models']['main_only']['by_suite']['Order']
    assert not order['coverage_complete'] and order['defined_trajectory_means']['pearson'] is None
    assert result['models']['main_only']['missing_episodes']==['Order/O01/a']
    policy['excluded_episodes'].append(dict(key='Order/O01/a',expected_samples=8))
    with pytest.raises(AssertionError):common_comparison(report,manifest,policy)
