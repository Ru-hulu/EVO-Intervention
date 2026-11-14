from fractions import Fraction
import pytest
from experiments.prm_five_groups_20261114.eval_common import sampling_indices,uniform_indices,trajectory_metrics,correlations


def test_sampling_matches_first_frame_at_each_timestamp_and_final():
    for rate in [Fraction(30),Fraction(25),Fraction(50,3),Fraction(20)]:
        actual=sampling_indices(123,rate)
        reference=[]; k=0
        for i in range(123):
            if i>=k*rate/5 or i==122:
                reference.append(i);k+=1
        assert actual==reference
    assert sampling_indices(1,30)==[0]
    with pytest.raises(ValueError):sampling_indices(20,5,10)


def test_cross_fps_integer_seconds_are_actual_inputs():
    for fps in [30,20,10,5,2]:
        assert set(range(0,303,30)).issubset(sampling_indices(303,30,fps))


def test_uniform_prefix_has_no_future_and_right_padding():
    import numpy as np
    for i in range(200):
        x=uniform_indices(i)
        assert len(x)==8 and min(x)==0 and max(x)==i
        expected=list(range(i+1))+[i]*(7-i) if i<7 else np.rint(np.linspace(0,i,8)).astype(int).tolist()
        assert x==expected


def test_official_vs_and_full_coverage_correlation_policy():
    t=[i/5 for i in range(9)];y=[i/8 for i in range(9)]
    r=trajectory_metrics(y,y,t)
    assert r['pearson']==pytest.approx(1) and r['spearman']==pytest.approx(1)
    assert r['VS']==1 and r['ER']==1 and r['C']==1
    r=trajectory_metrics([.5]*9,y,t)
    assert r['pearson'] is None and r['VS']==0 and r['C']==0
    r=trajectory_metrics([None]+y[1:],y,t)
    assert r['pearson'] is None and r['VS'] is None and r['invalid_predictions']==1
    r=trajectory_metrics([0,0,.2,.4,.4,.6,.7,.8,1],y,t)
    assert r['C']==pytest.approx(.75) and r['VS']==pytest.approx(.75)


def test_spearman_uses_mean_ranks_for_ties():
    p,s,reason=correlations([1,1,2,3],[3,2,2,1])
    assert reason is None and p==pytest.approx(-.8528028654224417)
    assert s==pytest.approx(-5/6)


def test_overlap_guard_distinguishes_reclaimable_cache_from_oom_risk():
    from experiments.prm_five_groups_20261114.overlap_guard import pause_reasons
    sample={'gpus':{0:{'used_mib':22000,'total_mib':81920,'temperature':60}},
            'memory_limit':460*2**30,'memory_usage':459*2**30,
            'resident_and_shmem':22*2**30,'memory_working_set':140*2**30,
            'under_oom':0,'oom_kill':0,'disk_free':500*2**30}
    assert pause_reasons(sample,0,0)==[]
    sample['gpus'][0]['used_mib']=70000
    assert 'less_than_16_GiB_GPU_headroom' in pause_reasons(sample,0,0)
    sample['gpus'][0]['used_mib']=22000;sample['oom_kill']=1
    assert 'container_OOM_requires_inspection' in pause_reasons(sample,0,0)
    sample['oom_kill']=0;sample['resident_and_shmem']=350*2**30
    assert 'container_resident_memory_high' in pause_reasons(sample,0,0)
