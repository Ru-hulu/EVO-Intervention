import numpy as np
import pytest
from vla_fail.acc_batch import describe_rows
from vla_fail.acc_batch import rank_records
from vla_fail.acc_batch import score_episode


def test_acc_only_matches_hand_calculation_and_has_no_llmd():
    # Stationary chunks: every axis differs by 1 mm; scale floor is 10 mm.
    chunks = np.zeros((3, 5, 3))
    chunks[1] = 0.001
    chunks[2] = 0.002
    rows = score_episode("demo/0", chunks, chunks, np.array([0, 2, 4]), np.array([0, 1, 2]))
    assert rows[0]["acc_max"] is None
    assert rows[1]["mae_mm_left"] == pytest.approx(1)
    assert rows[1]["acc_raw_left"] == pytest.approx(0.1)
    assert rows[1]["acc_max"] == pytest.approx(0.01)
    assert rows[2]["acc_max"] == pytest.approx(0.019)
    assert "llmd" not in rows[1]
    assert describe_rows(rows)["acc_p95"] == pytest.approx(0.01855)


def test_acc_compares_overlapping_future_positions():
    chunks = np.stack([np.broadcast_to(np.arange(5)[:, None] + offset, (5, 3)) for offset in (0, 2, 4)])
    rows = score_episode("aligned", chunks, chunks, np.array([0, 2, 4]), np.array([0, 1, 2]))
    assert rows[1]["acc_max"] == rows[2]["acc_max"] == 0
    with pytest.raises(ValueError, match="overlap"):
        score_episode("invalid", chunks, chunks, np.array([0, 5, 10]), np.array([0, 1, 2]))


def test_ranking_p95_not_longest_duration_or_single_maximum():
    records = [
        {"episode_id": str(i), "acc_p95": score, "acc_median": score / 2, "duration_s": duration, "acc_max": maximum}
        for i, (score, duration, maximum) in enumerate([(1, 1000, 100), (2, 30, 3), (3, 300, 4), (4, 100, 4)])
    ]
    ordered, selected = rank_records(list(reversed(records)))
    assert [r["episode_id"] for r in ordered] == ["0", "1", "2", "3"]
    assert [r["episode_id"] for r in selected["lowest"]] == ["0", "1"]
    assert [r["episode_id"] for r in selected["highest"]] == ["3", "2"]
    with pytest.raises(ValueError, match="unique"):
        rank_records([records[0]] * 4)
