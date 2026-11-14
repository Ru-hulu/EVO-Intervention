import copy
import numpy as np
import pytest
from experiments.prm_five_groups_20261114.recover_pour_labels import first_visit_mask,progress_from_deltas


def example():
    def stage(action,target,occurrence,start,end,index):
        return dict(action=action,target_position=target if action=='pour' else None,
                    after_pour_target=target if action=='put_down' else None,
                    target_occurrence=occurrence,source_start_frame=start,source_end_frame=end,
                    outcome='completed',segment_index=index)
    return {'frame_interval':'inclusive','frame_index_base':0,'total_frames':15,'stages':[
        stage('pour','left',1,1,2,1),stage('put_down','left',1,3,3,2),
        stage('pour','left',2,4,5,3),stage('put_down','left',2,6,6,4),
        stage('pour','middle',1,8,9,5),stage('pour','right',1,10,11,6),
        stage('put_down','right',1,12,13,7)],'idle_intervals':[
        {'source_start_frame':0,'source_end_frame':0},
        {'source_start_frame':7,'source_end_frame':7},
        {'source_start_frame':14,'source_end_frame':14}]}


def test_repeat_and_idle_hold_progress_even_with_large_visual_motion():
    a=example();raw=np.ones(15);raw[0]=0;raw[4:8]=10000
    effective,pt,events=progress_from_deltas(raw,a)
    assert pt[0]==0 and pt[-1]==1 and pt[3]==pytest.approx(1/3)
    assert np.array_equal(pt[3:8],np.full(5,1/3))
    assert not effective[4:8].any() and not effective[-1]
    assert pt[9]==pytest.approx(5/9)  # Direct cup transition needs no invented put-down.
    assert len(events)==7


@pytest.mark.parametrize('corrupt',['overlap','occurrence','wrong_put_down','missing_idle','missing_cup'])
def test_reject_inconsistent_annotations(corrupt):
    a=copy.deepcopy(example())
    if corrupt=='overlap':a['stages'][1]['source_start_frame']=2
    elif corrupt=='occurrence':a['stages'][2]['target_occurrence']=1
    elif corrupt=='wrong_put_down':a['stages'][1]['after_pour_target']='right'
    elif corrupt=='missing_idle':a['idle_intervals'].pop(1)
    else:a['stages'][5]['target_position']='left'
    with pytest.raises(ValueError):first_visit_mask(a)


def test_zero_effective_delta_has_no_time_based_fallback():
    a=example();raw=np.zeros(15);raw[4:8]=1000
    with pytest.raises(ValueError):progress_from_deltas(raw,a)
