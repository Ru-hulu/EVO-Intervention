"""Offline adapter contracts; optional LeRobot/Torch runtime, no checkpoint."""

import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest

# Optional runtime is checked before importing the standalone replay adapter.
# ruff: noqa: E402, SLF001
torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
path = Path(__file__).resolve().parents[3] / "scripts/replay_lerobot_vla_fail.py"
spec = importlib.util.spec_from_file_location("replay_adapter_test", path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_fixed_prior_capture_preserves_actions_and_cleans_up():
    class Core:
        def __init__(self):
            self.action_out_proj = torch.nn.Linear(4, 2)
            self.calls = 0

        def denoise_step(self, *, prefix_pad_masks, past_key_values, x_t, timestep):
            self.calls += 1
            return self.action_out_proj(torch.cat([x_t, x_t], dim=-1))

    class Policy:
        config = types.SimpleNamespace(num_inference_steps=2)

        def __init__(self):
            self.model = Core()

        def predict_action_chunk(self, batch, *, noise, num_steps=None):
            steps = self.config.num_inference_steps if num_steps is None else num_steps
            x = noise
            for i in range(steps):
                x = (
                    x
                    - self.model.denoise_step(
                        prefix_pad_masks=None, past_key_values=None, x_t=x, timestep=torch.tensor([1 - i / steps])
                    )
                    / steps
                )
            return x

    policy = Policy()
    original_method = policy.model.denoise_step.__func__
    noise = torch.zeros(1, 3, 2)
    fixed = torch.ones(1, 3, 2)
    reference = policy.predict_action_chunk({}, noise=noise)
    policy.model.calls = 0
    rng = torch.random.get_rng_state().clone()
    features, result = adapter.capture_features(policy, {}, fixed, noise=noise)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    np.testing.assert_array_equal(features, np.ones((3, 4)))
    assert policy.model.calls == 3
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not policy.model.action_out_proj._forward_pre_hooks
    assert policy.model.denoise_step.__func__ is original_method
    with pytest.raises(RuntimeError, match="finite"):
        adapter.capture_features(policy, {}, fixed * float("nan"), noise=noise)
    assert not policy.model.action_out_proj._forward_pre_hooks
    assert policy.model.denoise_step.__func__ is original_method


def test_piper_units_and_arm_indices():
    class FK:
        def __init__(self):
            self.seen = []

        def CalFK(self, joints):  # noqa: N802 - SDK API name
            self.seen.append(joints)
            return [[1000, 2000, 3000, 0, 0, 0]]

    fk = FK()
    actions = np.zeros((2, 14))
    actions[:, 0] = 180
    actions[:, 7] = 90
    result = adapter.xyz_from_joint_degrees(actions, fk)
    assert fk.seen[0][0] == pytest.approx(np.pi)
    assert fk.seen[2][0] == pytest.approx(np.pi / 2)
    np.testing.assert_array_equal(result["positions_left"], [[1, 2, 3], [1, 2, 3]])
    with pytest.raises(ValueError, match="14"):
        adapter.xyz_from_joint_degrees(np.zeros((2, 7)), fk)
