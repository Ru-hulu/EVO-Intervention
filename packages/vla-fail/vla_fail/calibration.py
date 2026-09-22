"""Constant, trajectory-level split-conformal thresholds on success rollouts.

The local paper names a constant conformal band but delegates details to FIPER.
This module explicitly chooses episode maxima as nonconformity scores. It is a
documented calibration implementation, not a claim of identical author code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ._validation import array
from ._validation import scalar
from .results import JsonResult


@dataclass(frozen=True)
class ThresholdCalibration(JsonResult):
    threshold: float
    raw_quantile: float
    alpha: float
    n_episodes: int
    order_statistic: int  # One-based ceil((n+1)*(1-alpha)).
    episode_maxima: list[float]


def calibrate_threshold(successful_episode_scores, alpha: float = 0.05) -> ThresholdCalibration:
    """Each element must be one successful episode's nonnegative score series.

    Use an independent held-out set. For ACC, pass smoothed scores with identical
    EMA initialization/reset as deployment; omit the unavailable first step.
    Exchangeability of success episodes is needed for the coverage statement.
    Scores need not be independent within an episode. This is NOT failure recall
    calibration and gives no conditional guarantee on an OOD rollout.

    The next representable float above the empirical quantile makes the paper's
    >= alarm equivalent to > quantile, including ties. Too few episodes raises
    instead of silently using a finite threshold without the requested coverage.
    """
    level = scalar(alpha, "alpha", strict=True)
    if level >= 1:
        raise ValueError("alpha must be in (0,1)")
    maxima = []
    for index, episode in enumerate(successful_episode_scores):
        scores = array(episode, f"episode {index} scores", 1)
        if np.any(scores < 0):
            raise ValueError("LLMD/ACC calibration scores must be nonnegative")
        maxima.append(float(scores.max()))
    count = len(maxima)
    if count == 0:
        raise ValueError("At least one successful calibration episode is required")
    rank = math.ceil((count + 1) * (1 - level))
    if rank > count:
        raise ValueError(
            f"{count} episodes cannot yield a finite conformal threshold at alpha={level}; "
            "collect more independent successful episodes or choose a larger alpha"
        )
    quantile = float(np.partition(np.asarray(maxima), rank - 1)[rank - 1])
    threshold = float(np.nextafter(quantile, np.inf))
    if not np.isfinite(threshold):
        raise ValueError("Calibration threshold overflowed")
    return ThresholdCalibration(threshold, quantile, level, count, rank, maxima)
