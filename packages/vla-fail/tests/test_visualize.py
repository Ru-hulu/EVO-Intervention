import json

import numpy as np
import pytest
from vla_fail.llmd import LLMDDetector
from vla_fail.visualize import evaluate
from vla_fail.visualize import render


@pytest.fixture
def inputs():
    rng = np.random.default_rng(31)
    fixed = np.zeros((3, 2), dtype=np.float32)
    detector = LLMDDetector(fixed_noise=fixed, metadata={"test_model": "tiny"}).fit(rng.normal(size=(20, 3, 4)))
    positions = np.zeros((3, 3, 3))
    for i in range(3):
        positions[i, :, 0] = np.arange(3) + i
    episode = {
        "episode_id": "failure-0",
        "metadata": detector.metadata,
        "fixed_noise": fixed,
        "features": rng.normal(size=(3, 3, 4)),
        "positions_left": positions,
        "positions_right": positions.copy(),
        "executed_steps": np.array([0, 1, 1]),
        "timestamps": np.array([0, 0.1, 0.2]),
        "frame_indices": np.arange(3),
        "success": np.array(False),  # noqa: FBT003 - explicit scalar failure label
    }
    episode["positions_right"][1] += 0.1
    return episode, detector


def test_per_arm_alignment_and_unknown_thresholds(inputs):
    episode, detector = inputs
    rows, tokens, arms = evaluate(episode, detector)
    assert arms == ["left", "right"]
    assert tokens.shape == (3, 3)
    assert all(row["need_intervention"] is None for row in rows)
    assert rows[0]["acc_left"] is None
    assert rows[1]["acc_left"] == 0
    assert rows[1]["acc_right"] > 0
    assert rows[1]["acc_max"] == rows[1]["acc_right"]


def test_rejects_invalid_alignment(inputs):
    episode, detector = inputs
    episode["timestamps"][1] = -1
    with pytest.raises(ValueError, match="increasing"):
        evaluate(episode, detector)
    episode["timestamps"] = np.array([0, 0.1, 0.2])
    episode["positions_left"] = np.zeros((3, 3, 14))
    with pytest.raises(ValueError, match="positions_left"):
        evaluate(episode, detector)


def test_plot_exports_do_not_invent_failure_onset(inputs, tmp_path):
    pytest.importorskip("matplotlib")
    episode, detector = inputs
    stats = tmp_path / "stats.npz"
    detector.save(stats)
    path = tmp_path / "episode.npz"
    np.savez(path, **{**episode, "metadata": np.array(json.dumps(episode["metadata"]))})
    summary = render(path, stats, tmp_path / "output")
    assert summary["failure_time_s"] is None
    assert summary["llmd_threshold"] is None
    assert summary["acc_threshold"] is None
    for name in ["scores.png", "scores.svg", "scores.csv", "summary.json"]:
        assert (tmp_path / "output" / name).stat().st_size > 100
