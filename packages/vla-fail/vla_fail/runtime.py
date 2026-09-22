"""Stateful, single-environment adapter for post-transform policy actions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .acc import ACCDetector
from .detector import VLAFAILDetector
from .llmd import LLMDDetector

FEATURE_METADATA = {"feature_source": "pi05_pytorch.action_out_proj.input", "flow_time": 1.0}


class PolicyMonitor:
    """One instance per environment. ACC accepts absolute Cartesian XYZ only.

    Joint-space actions require forward kinematics outside this adapter. No
    inference about the meaning of the first three action dimensions is made.
    """

    def __init__(self, *, llmd=None, acc=None, position_indices=None):
        self.detector = VLAFAILDetector(llmd=llmd, acc=acc)
        if llmd is not None:
            if not llmd.fitted or llmd.fixed_noise is None:
                raise ValueError("Online LLMD requires fitted statistics and saved fixed_noise")
            for key, value in FEATURE_METADATA.items():
                if llmd.metadata.get(key) != value:
                    raise ValueError(f"LLMD feature metadata mismatch: {key}")
        if acc is not None:
            if (
                not isinstance(position_indices, list | tuple)
                or len(position_indices) != 3
                or any(type(i) is not int or i < 0 for i in position_indices)
                or len(set(position_indices)) != 3
            ):
                raise ValueError("ACC requires three distinct absolute Cartesian position_indices")
        elif position_indices is not None:
            raise ValueError("position_indices require ACC")
        self.position_indices = position_indices
        self.reset()

    @classmethod
    def load(cls, *, stats=None, acc_config=None, expected_metadata=None):
        llmd = None if stats is None else LLMDDetector.load(stats, expected_metadata=expected_metadata)
        acc, indices = None, None
        if acc_config is not None:
            config = json.loads(Path(acc_config).read_text(encoding="utf-8"))
            if config.get("action_representation") != "absolute_xyz":
                raise ValueError("Only explicit absolute_xyz ACC is supported; convert joint/delta actions first")
            indices = config["position_indices"]
            acc = ACCDetector(
                alpha=config.get("ema_alpha", 0.9), v_min=config.get("v_min", 0.001), threshold=config.get("threshold")
            )
        return cls(llmd=llmd, acc=acc, position_indices=indices)

    @property
    def fixed_noise(self):
        return None if self.detector.llmd is None else self.detector.llmd.fixed_noise

    def reset(self):
        self.detector.reset()
        self._episode_id = None
        self._has_previous = False
        self._previous_horizon = None

    def begin(self, context):
        """Validate before inference; return whether the semantic window must reset."""
        if not isinstance(context, dict):
            raise ValueError("obs['vla_fail'] must be an object")
        episode_id = context.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Monitoring requires a nonempty vla_fail.episode_id")
        reset = context.get("reset", False)
        if type(reset) is not bool:
            raise ValueError("vla_fail.reset must be boolean")
        changed = reset or episode_id != self._episode_id
        if self.detector.acc is not None and self._has_previous and not changed:
            steps = context.get("executed_steps")
            if type(steps) is not int or not 0 < steps < self._previous_horizon:
                raise ValueError(
                    "executed_steps must be the actual 0 < R < H since the preceding chunk; "
                    "reset after takeover or a nonoverlapping chunk"
                )
        if changed:
            self.reset()
            self._episode_id = episode_id
        return changed

    def step(self, *, features, actions, context):
        positions, steps = None, None
        if self.detector.acc is not None:
            actions = np.asarray(actions)
            if actions.ndim != 2 or max(self.position_indices) >= actions.shape[1]:
                raise ValueError("Policy actions do not contain the configured absolute XYZ indices")
            positions = actions[:, self.position_indices]
            steps = context.get("executed_steps") if self._has_previous else None
        result = self.detector.step(tokens=features, positions=positions, executed_steps=steps)
        self._has_previous = True
        if positions is not None:
            self._previous_horizon = len(positions)
        return result.to_dict()
