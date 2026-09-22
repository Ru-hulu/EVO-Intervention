"""Absolute end-effector chunk consistency (paper Section 3.2)."""

# Dataclass positional constructors mirror the result schema.
# ruff: noqa: FBT003
from __future__ import annotations

import numpy as np

from ._validation import ThresholdMixin
from ._validation import array
from ._validation import integer
from ._validation import scalar
from .results import ACCResult


def _positions(value, name: str) -> np.ndarray:
    positions = array(value, name, 2)
    if positions.shape[1] != 3:
        raise ValueError(f"{name} must be [H,3] absolute end-effector XYZ, not raw full actions")
    return positions


def delta_xyz_to_absolute(deltas, start_xyz) -> np.ndarray:
    """ONLY for successive Cartesian increments in one fixed coordinate frame.

    p[i] = start_xyz + sum(deltas[:i+1]). Not for joint deltas, local tool-frame
    deltas, normalized actions or offsets each relative to the same initial pose.
    Old/new chunks must each use their own actual start pose.
    """
    delta = _positions(deltas, "deltas")
    start = array(start_xyz, "start_xyz", 1)
    if start.shape != (3,):
        raise ValueError("start_xyz must have shape [3]")
    return _positions(start + np.cumsum(delta, axis=0), "absolute positions")


class ACCDetector(ThresholdMixin):
    """One instance per environment/episode; reset after handover or reset.

    Defaults to the paper's simulation settings alpha=.9 and v_min=.001.
    Zero-initialized EMA is an explicit implementation choice (unspecified in
    the paper). score_pair is stateless; update maintains one stream's history.
    """

    def __init__(self, alpha: float = 0.9, v_min: float = 1e-3, threshold: float | None = None):
        self.alpha = scalar(alpha, "alpha")
        if self.alpha >= 1:
            raise ValueError("alpha must be in [0,1)")
        self.v_min = scalar(v_min, "v_min", strict=True)
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._previous = None
        self._smoothed = 0.0

    def score_pair(
        self, previous_positions, current_positions, executed_steps: int, *, previous_smoothed_score: float = 0.0
    ) -> ACCResult:
        """Compare previous[R:H] with current[:H-R], then update the supplied EMA.

        Both chunks must have the same H. R is the number actually executed
        between queries, not an inference counter or elapsed wall-clock time.
        All available overlap is used, exactly as in the main paper formula.
        """
        previous = _positions(previous_positions, "previous_positions")
        current = _positions(current_positions, "current_positions")
        if previous.shape != current.shape:
            raise ValueError("Both action chunks must have the same [H,3] shape")
        steps = integer(executed_steps, "executed_steps")
        if steps >= previous.shape[0]:
            raise ValueError("ACC needs 1 <= executed_steps < H; no overlap remains")
        smoothed = scalar(previous_smoothed_score, "previous_smoothed_score")
        overlap = previous.shape[0] - steps
        old, new = previous[steps:], current[:overlap]
        scale = np.maximum(np.ptp(new, axis=0), self.v_min)
        mae = np.abs(old - new).mean(axis=0)
        normalized = mae / scale
        raw_score = float(normalized.mean())
        score = self.alpha * smoothed + (1.0 - self.alpha) * raw_score
        if not np.isfinite(score) or not np.isfinite(raw_score) or not np.isfinite(scale).all():
            raise ValueError("ACC calculation overflowed; check coordinate scale")
        return ACCResult(
            True,
            score,
            raw_score,
            mae.tolist(),
            scale.tolist(),
            normalized.tolist(),
            overlap,
            self.threshold,
            None if self.threshold is None else score >= self.threshold,
        )

    def update(self, positions, executed_steps: int | None = None) -> ACCResult:
        """First call caches a chunk and returns ready=False, triggered=None.

        Invalid calls do not change history. Set executed_steps on every later
        call; call reset if a chunk was replaced by another controller.
        """
        current = _positions(positions, "positions")
        if current.shape[0] < 2:
            raise ValueError("ACC needs action chunks with H >= 2")
        if self._previous is None:
            if executed_steps is not None:
                steps = integer(executed_steps, "executed_steps")
                if steps >= current.shape[0]:
                    raise ValueError("ACC needs executed_steps < H")
            result = ACCResult(False, None, None, None, None, None, 0, self.threshold, None)
        else:
            if executed_steps is None:
                raise ValueError("executed_steps is required after the first chunk")
            result = self.score_pair(self._previous, current, executed_steps, previous_smoothed_score=self._smoothed)
        self._previous = current.copy()
        if result.ready:
            self._smoothed = result.score
        return result
