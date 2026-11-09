import numpy as np
import pytest

from mem_lite.datasets.prepare_proprimem import whole_video_progress, split_videos


def test_whole_video_pt_tracks_visual_change_without_stage_annotations():
    delta = [0, 1, 0, 3, 0]
    expected = [0, 0.25, 0.25, 1, 1]
    np.testing.assert_array_equal(whole_video_progress(delta), expected)
    np.testing.assert_array_equal(whole_video_progress(np.array(delta)*7), expected)
    # 总视觉变化很不均匀，不应退化为按帧号线性插值。
    assert not np.allclose(whole_video_progress(delta), np.linspace(0,1,5))


@pytest.mark.parametrize('delta', [[0,0,0], [0,1e-12], [1,2], [0,-1,2], [0,float('nan')]])
def test_undefined_pt_requires_review_instead_of_synthetic_fallback(delta):
    with pytest.raises(ValueError):
        whole_video_progress(delta)


def test_video_split_is_reproducible_and_independent_of_input_order():
    ids = [f'episode_{i:06d}' for i in range(30)]
    split = split_videos(ids,'task')
    assert split == split_videos(ids[::-1],'task')
    assert list(split.values()).count('train') == 24
    assert list(split.values()).count('validation') == 6
    assert set(split) == set(ids)
