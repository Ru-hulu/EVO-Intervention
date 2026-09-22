import json

import numpy as np
import pytest
from vla_fail.acc import ACCDetector
from vla_fail.cli import calibrate
from vla_fail.cli import fit
from vla_fail.cli import load_episode
from vla_fail.cli import replay
from vla_fail.runtime import PolicyMonitor


def test_monitor_alignment_reset_and_unknown():
    monitor = PolicyMonitor(acc=ACCDetector(threshold=0.1), position_indices=[0, 1, 2])
    context = {"episode_id": "a"}
    assert monitor.begin(context)
    positions = np.arange(18).reshape(6, 3)
    first = monitor.step(features=None, actions=positions, context=context)
    assert first["need_intervention"] is None
    with pytest.raises(ValueError, match="executed_steps"):
        monitor.begin(context)
    context["executed_steps"] = 2
    assert not monitor.begin(context)
    second = monitor.step(features=None, actions=positions + 6, context=context)
    assert second["acc"]["score"] == 0
    assert second["need_intervention"] is False
    assert monitor.begin({"episode_id": "b"})
    assert monitor.step(features=None, actions=positions, context={"episode_id": "b"})["acc"]["ready"] is False


def test_monitor_rejects_implicit_joint_coordinates(tmp_path):
    config = tmp_path / "acc.json"
    config.write_text(json.dumps({"action_representation": "joints", "position_indices": [0, 1, 2]}))
    with pytest.raises(ValueError, match="absolute_xyz"):
        PolicyMonitor.load(acc_config=config)
    with pytest.raises(ValueError, match="distinct"):
        PolicyMonitor(acc=ACCDetector(), position_indices=[0, 0, 1])


def episode(path, eid, *, success=True, noise=None):
    np.savez(
        path,
        features=np.arange(48).reshape(4, 3, 4) / 100,
        positions=np.arange(36).reshape(4, 3, 3) / 100,
        executed_steps=np.array([0, 1, 1, 1]),
        episode_id=np.array(eid),
        success=np.array(success),
        metadata=np.array(json.dumps({"checkpoint": "test"})),
        fixed_noise=np.ones((3, 2)) if noise is None else noise,
    )


def test_offline_fit_calibration_and_replay(tmp_path):
    train = tmp_path / "train.npz"
    episode(train, "train")
    stats = tmp_path / "stats.npz"
    fit([train], stats)
    paths = []
    for i in range(9):
        path = tmp_path / f"cal_{i}.npz"
        episode(path, f"cal-{i}")
        paths.append(path)
    report = calibrate(paths, tmp_path / "calibrated", stats=stats, acc_config={}, alpha=0.2)
    assert report["signals"]["llmd"]["alpha"] == 0.1
    from vla_fail.llmd import LLMDDetector

    detector = LLMDDetector.load(tmp_path / "calibrated" / "llmd.npz")
    rows = replay(load_episode(paths[0]), llmd=detector, acc_config={"threshold": 1e9})
    assert rows[0]["need_intervention"] is None
    assert all(row["need_intervention"] is False for row in rows[1:])
    with pytest.raises(ValueError, match="disjoint"):
        calibrate([train], tmp_path / "bad", stats=stats)
    episode(paths[0], "failure", success=False)
    with pytest.raises(ValueError, match="successful"):
        calibrate(paths, tmp_path / "bad", stats=stats)
    episode(paths[0], "wrong-noise", noise=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="fixed_noise"):
        replay(load_episode(paths[0]), llmd=detector)
