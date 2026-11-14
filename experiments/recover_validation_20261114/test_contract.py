import copy
import pytest
from .common import select,frame_inputs,MODELS
from .worker import validate_prefix
from .report import summary


def test_shared_holdout_deterministic_and_disjoint():
    tasks=[dict(task_id=f'R{i:02d}',train=['a','b'],val=['c','d','e']) for i in range(1,6)]
    assert select(tasks)==select(list(reversed(tasks)))
    assert all(len(v)==2 and not set(v)&{'a','b'} for v in select(tasks).values())
    tasks[0]['val'][0]='a'
    with pytest.raises(AssertionError):select(tasks)
    assert 'vlac-2b' not in MODELS and len(MODELS)==5


def test_gt_does_not_enter_request():
    row=dict(key='Recover/R01/x',task_id='R01',sample_id='x',task_prompt='goal',frame_indices=[0,6],target_pt=[0.,1.])
    actual=frame_inputs(row);row['target_pt']=[99.,99.]
    assert frame_inputs(row)==actual and set(actual)=={'key','task','frames'}


def test_resume_rejects_changed_truth_frames_or_weights():
    row=dict(key='x',frame_indices=[0,6],target_pt=[0.,.8])
    result=dict(key='x',inputs_sha256='a',checkpoint_sha256='b',frame_indices=[0],target_pt=[0.],predictions=[.1],raw_responses=['raw'])
    assert validate_prefix(result,row,'a','b')==1
    for key,value in [('target_pt',[.2]),('frame_indices',[1]),('checkpoint_sha256','other')]:
        bad=copy.deepcopy(result);bad[key]=value
        with pytest.raises(AssertionError):validate_prefix(bad,row,'a','b')


def test_no_partial_suite_mean_or_null_substitution():
    records=[dict(pearson=.5,spearman=.4,invalid_predictions=0),
             dict(pearson=None,spearman=None,correlation_undefined_reason='constant_prediction',invalid_predictions=0)]
    assert summary(records,10)['metrics']['pearson']['mean'] is None
    full=summary(records,2)['metrics']['pearson']
    assert full['mean']==.5 and full['defined_episodes']==full['undefined_episodes']==1


def test_contiguous_repeat_platform_keeps_one_drift_anchor():
    from .events import merge_platforms
    from experiments.prm_five_groups_20261114.recover_metrics import score_recover
    intervals=[dict(start_frame=1,end_frame=2,cause='repeated_action'),dict(start_frame=3,end_frame=4,cause='repeated_action'),
               dict(start_frame=7,end_frame=8,cause='repeated_action')]
    merged=merge_platforms(intervals)
    assert [(s['start_frame'],s['end_frame']) for s in merged]==[(1,4),(7,8)]
    score=score_recover([0.,.2,.3,.4,.5],[0.,.2,.2,.2,.2],list(range(5)),list(range(5)),5,
        dict(success=True,trr_role='normal',failure_onset_frame=None,abnormal_plateaus=merged[:1]))
    assert score['abnormal_plateau_drift_mae']==pytest.approx(.15)


def test_final_forest_constant_lr_has_no_scheduler():
    from .worker import check_forest_checkpoint
    ck=dict(epoch=1,next_batch=0,rng_by_rank=[{},{}],optimizer={'state':{1:'optimizer'}},step=319,scheduler=None,
        config=dict(lr_schedule='constant',run_protocol=dict(train_episodes=10,suite='Recover',fps=5)))
    check_forest_checkpoint(ck,dict(optimizer_steps=319))
    ck['config']['lr_schedule']='cosine'
    with pytest.raises((AssertionError,TypeError)):check_forest_checkpoint(ck,dict(optimizer_steps=319))
