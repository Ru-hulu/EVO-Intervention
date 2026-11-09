"""No GPUs: verify that failure never advances the automatic training chain."""
from unittest.mock import patch

import pytest

import autostart


@pytest.mark.parametrize("failure", ["cache", "smoke", "train", None])
def test_automatic_chain_requires_prior_stage_success(failure):
    events = []

    def action(stage):
        events.append(stage)
        if stage == failure:
            raise RuntimeError(stage + " failed")

    if failure is None:
        autostart.execute_chain(lambda: action("cache"), action, lambda: action("verify"))
        assert events == ["cache", "smoke", "train", "verify"]
    else:
        with pytest.raises(RuntimeError, match=failure + " failed"):
            autostart.execute_chain(lambda: action("cache"), action, lambda: action("verify"))
        assert events == ["cache", "smoke", "train"][:["cache", "smoke", "train"].index(failure) + 1]


def test_completed_caches_do_not_require_a_live_controller():
    with patch.object(autostart.os, "kill", side_effect=AssertionError("must not check old PID")):
        assert autostart.cache_ready({"phase": "completed"})


def test_failed_cache_and_dead_controller_stop_chain():
    with pytest.raises(RuntimeError, match="Cache generation failed"):
        autostart.cache_ready({"phase": "failed", "error": "fixture failure"})
    with patch.object(autostart.os, "kill", side_effect=ProcessLookupError):
        with pytest.raises(RuntimeError, match="exited before completing"):
            autostart.cache_ready({"phase": "building_base_cache", "pid": 123})


def test_running_cache_waits():
    with patch.object(autostart.os, "kill", return_value=None):
        assert not autostart.cache_ready({"phase": "building_augmented_cache", "pid": 123})
