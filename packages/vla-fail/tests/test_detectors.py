import numpy as np
import pytest
from vla_fail import ACCDetector
from vla_fail import LLMDDetector
from vla_fail import VLAFAILDetector
from vla_fail import calibrate_threshold
from vla_fail import delta_xyz_to_absolute
from vla_fail import make_fixed_noise


@pytest.fixture
def training():
    return np.array([[[1, 8], [4, 3]], [[1, 12], [4, 7]], [[3, 8], [6, 3]], [[3, 12], [6, 7]]], dtype=float)


@pytest.fixture
def detector(training):
    model = LLMDDetector().fit(training)
    model.threshold = 10
    return model


@pytest.fixture
def chunks():
    old = np.array([[0.1, 0.2, 0.1], [0.2, 0.2, 0.1], [0.3, 0.2, 0.1], [0.4, 0.2, 0.1], [0.5, 0.2, 0.1]])
    bad = np.array([[0.1, 0.24, 0.1], [0.2, 0.24, 0.1], [0.3, 0.24, 0.1], [0.4, 0.24, 0.1], [0.5, 0.24, 0.1]])
    return old, bad


def test_llmd_hand_calculation_and_population_covariance(detector):
    result = detector.score([[3, 12], [8, 9]])
    expected = [1 / 1.000001 + 4 / 4.000001, 9 / 1.000001 + 16 / 4.000001]
    np.testing.assert_allclose(result.per_token_scores, expected, rtol=1e-12)
    assert result.score == pytest.approx(expected[1])  # Max, not mean or square root.
    assert result.worst_token == 1
    assert result.triggered is True
    np.testing.assert_equal(detector.mean, [[2, 10], [5, 5]])


def test_full_covariance_detects_unusual_direction():
    # Covariance [[1,.9],[.9,1]]: same Euclidean norm, very different likelihood.
    square = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)
    transform = np.array([[1, 0], [0.9, np.sqrt(0.19)]])
    model = LLMDDetector().fit((square @ transform.T)[:, None, :])
    assert model.score([[1, 1]]).score == pytest.approx(2 / (1.9 + 1e-6))
    assert model.score([[1, -1]]).score == pytest.approx(2 / (0.1 + 1e-6))


def test_streaming_statistics_match_single_batch():
    rng = np.random.default_rng(71)
    data = rng.normal(size=(37, 3, 7)) + 1e5
    whole = LLMDDetector().fit(data)
    stream = LLMDDetector()
    for batch in np.array_split(data, 8):
        stream.partial_fit(batch)
    stream.finalize()
    np.testing.assert_allclose(stream.mean, whole.mean, atol=1e-9)
    for sample in data[:3]:
        np.testing.assert_allclose(
            stream.score(sample).per_token_scores, whole.score(sample).per_token_scores, rtol=1e-9
        )
    assert stream.n_samples == 37


def test_rank_deficient_features_remain_finite():
    model = LLMDDetector().fit(np.ones((2, 3, 8)))
    assert model.score(np.ones((3, 8))).score == 0
    assert model.score(np.ones((3, 8)) * 2).score == pytest.approx(8e6)


def test_artifact_roundtrip_and_metadata(tmp_path, training):
    noise = make_fixed_noise((2, 7), seed=7)
    model = LLMDDetector(fixed_noise=noise, metadata={"checkpoint": "test", "task": "cup"}).fit(training)
    model.threshold = 10
    path = tmp_path / "model.npz"
    model.save(path)
    restored = LLMDDetector.load(path, expected_metadata={"task": "cup"})
    assert restored.score(training[0]).to_dict() == model.score(training[0]).to_dict()
    np.testing.assert_array_equal(restored.fixed_noise, noise)
    assert restored.n_samples == 4
    with pytest.raises(ValueError, match="metadata mismatch"):
        LLMDDetector.load(path, expected_metadata={"task": "kettle"})


def test_batch_and_refit_clear_stale_threshold(detector, training):
    assert len(detector.score_batch(training)) == 4
    detector.fit(training)
    assert detector.score(training[0]).triggered is None
    with pytest.raises(RuntimeError, match="finalized"):
        detector.partial_fit(training)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 3)),
        np.zeros((1, 2, 2)),
        [[np.nan, 1], [2, 3]],
        [[np.inf, 1], [2, 3]],
        [[1 + 2j, 1], [2, 3]],
        [["1", "2"], ["3", "4"]],
    ],
)
def test_invalid_tokens_fail_explicitly(detector, bad):
    with pytest.raises(ValueError, match="tokens"):
        detector.score(bad)


def test_unfitted_and_insufficient_statistics():
    with pytest.raises(RuntimeError):
        LLMDDetector().score([[0, 0]])
    with pytest.raises(ValueError, match="two observations"):
        LLMDDetector().fit(np.ones((1, 2, 2)))


def test_acc_hand_calculation(chunks):
    old, new = chunks
    acc = ACCDetector(v_min=0.01, threshold=0.5)
    result = acc.score_pair(old, new, 2, previous_smoothed_score=0.02)
    np.testing.assert_allclose(result.per_axis_mae, [0.2, 0.04, 0])
    np.testing.assert_allclose(result.per_axis_scale, [0.2, 0.01, 0.01])
    np.testing.assert_allclose(result.per_axis_normalized_error, [1, 4, 0])
    assert result.raw_score == pytest.approx(5 / 3)
    assert result.score == pytest.approx(0.18466666666666667)
    assert result.overlap == 3
    assert result.triggered is False


def test_acc_aligned_trajectory_has_zero_error(chunks):
    old, _ = chunks
    new = old.copy()
    new[:, 0] += 0.2
    # Old indices 2,3,4 coincide with new indices 0,1,2.
    assert ACCDetector().score_pair(old, new, 2).raw_score == pytest.approx(0, abs=1e-12)


def test_acc_does_not_use_unexecuted_new_tail_for_scale(chunks):
    old, new = chunks
    original = ACCDetector(v_min=0.01).score_pair(old, new, 2)
    new[3:] = 1e5
    modified = ACCDetector(v_min=0.01).score_pair(old, new, 2)
    assert original.to_dict() == modified.to_dict()


def test_acc_persistent_error_ema_crosses_threshold(chunks):
    old, new = chunks
    acc = ACCDetector(v_min=0.01, threshold=0.5)
    ema, flags = 0.02, []
    for _ in range(4):
        result = acc.score_pair(old, new, 2, previous_smoothed_score=ema)
        ema = result.score
        flags.append(result.triggered)
    assert flags == [False, False, False, True]
    assert ema == pytest.approx(0.5862886666666667)


def test_acc_stationary_and_one_step_overlap():
    chunk = np.ones((3, 3))
    acc = ACCDetector()
    result = acc.score_pair(chunk, chunk, 2)
    assert result.raw_score == 0
    assert result.overlap == 1
    np.testing.assert_equal(result.per_axis_scale, [0.001] * 3)


def test_acc_warmup_reset_copy_and_failed_call_atomicity(chunks):
    old, new = chunks
    original = old.copy()
    acc = ACCDetector(v_min=0.01)
    warmup = acc.update(old)
    assert not warmup.ready
    assert warmup.triggered is None
    assert warmup.score is None
    old[:] = 99  # The cached chunk is owned by the detector.
    with pytest.raises(ValueError, match="overlap"):
        acc.update(new, 5)
    result = acc.update(new, 2)
    assert result.raw_score == pytest.approx(acc.score_pair(original, new, 2).raw_score)
    acc.reset()
    assert not acc.update(new).ready


@pytest.mark.parametrize("steps", [None, 0, -1, 5, 6, 1.5, True])
def test_acc_rejects_invalid_execution_stride(chunks, steps):
    old, new = chunks
    acc = ACCDetector()
    acc.update(old)
    with pytest.raises(ValueError, match="executed_steps"):
        acc.update(new, steps)


def test_acc_requires_xyz_and_constant_horizon(chunks):
    old, new = chunks
    with pytest.raises(ValueError, match="XYZ"):
        ACCDetector().update(np.ones((5, 7)))
    with pytest.raises(ValueError, match="same"):
        ACCDetector().score_pair(old, new[:3], 2)


def test_cartesian_increment_conversion():
    np.testing.assert_allclose(
        delta_xyz_to_absolute([[0.1, 0, 0], [0.2, -0.1, 0]], [1, 2, 3]), [[1.1, 2, 3], [1.3, 1.9, 3]]
    )


def test_calibration_episode_max_not_pooled_steps_and_ties():
    episodes = [[float(i), 0] for i in range(1, 20)]
    report = calibrate_threshold(episodes, alpha=0.05)
    assert report.order_statistic == 19
    assert report.raw_quantile == 19
    assert report.threshold > 19
    assert np.nextafter(19.0, np.inf) >= report.threshold
    # Duplicating time steps does not turn a long episode into multiple samples.
    repeated = [episode * 100 for episode in episodes]
    assert calibrate_threshold(repeated).threshold == report.threshold
    zeros = calibrate_threshold([[0.0]] * 19)
    assert not (zeros.threshold <= 0.0)


@pytest.mark.parametrize(
    ("episodes", "alpha"),
    [([], 0.05), ([[1]] * 18, 0.05), ([[1]] * 20, 0.025), ([[1]] * 20, 0), ([[1]] * 20, 1), ([[]], 0.5), ([[-1]], 0.5)],
)
def test_invalid_calibration_is_not_silently_accepted(episodes, alpha):
    with pytest.raises(ValueError, match="episode|alpha|nonnegative"):
        calibrate_threshold(episodes, alpha)


def test_llmd_only_and_or_unknown_semantics(detector, chunks):
    old, new = chunks
    only = VLAFAILDetector(llmd=detector)
    assert only.step(tokens=detector.mean).need_intervention is False
    combo = VLAFAILDetector(llmd=detector, acc=ACCDetector(v_min=0.01, threshold=0.1))
    first = combo.step(tokens=detector.mean, positions=old)
    assert first.need_intervention is None
    assert not first.decision_complete
    assert first.unavailable == ["acc_history_missing"]
    second = combo.step(tokens=detector.mean, positions=new, executed_steps=2)
    assert second.need_intervention is True
    assert second.reasons == ["acc_action_inconsistency"]
    combo.reset()
    alarm = combo.step(tokens=[[3, 12], [8, 9]], positions=old)
    assert alarm.need_intervention is True
    assert not alarm.decision_complete
    assert alarm.reasons == ["llmd_feature_ood"]


def test_uncalibrated_signal_is_unknown(detector):
    detector.threshold = None
    result = VLAFAILDetector(llmd=detector).step(tokens=[[3, 12], [8, 9]])
    assert result.need_intervention is None
    assert result.unavailable == ["llmd_threshold_missing"]


def test_joint_invalid_features_do_not_advance_acc(detector, chunks):
    old, new = chunks
    combined = VLAFAILDetector(llmd=detector, acc=ACCDetector(v_min=0.01, threshold=0.5))
    combined.step(tokens=detector.mean, positions=old)
    with pytest.raises(ValueError, match="tokens"):
        combined.step(tokens=np.zeros((1, 1)), positions=new, executed_steps=2)
    result = combined.step(tokens=detector.mean, positions=new, executed_steps=2)
    assert result.acc.score == pytest.approx(1 / 6)


@pytest.mark.parametrize("kwargs", [{"alpha": 1}, {"alpha": -1}, {"v_min": 0}, {"threshold": np.nan}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError, match="alpha|v_min|threshold"):
        ACCDetector(**kwargs)
