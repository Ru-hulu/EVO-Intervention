"""Token-wise squared Mahalanobis distance (paper Section 3.1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from ._validation import ThresholdMixin
from ._validation import array
from ._validation import integer
from ._validation import scalar
from .results import LLMDResult


def make_fixed_noise(action_shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    """Sample once; reuse this array for fitting, calibration and deployment.

    Shape is the model's full [H, action_dim], NOT necessarily [H, 3].
    The caller feeds this to the model at the pure-noise endpoint.
    """
    if len(action_shape) != 2:
        raise ValueError("action_shape must be (H, action_dim)")
    shape = tuple(integer(size, "action_shape dimension") for size in action_shape)
    return np.random.default_rng(integer(seed, "seed", 0)).standard_normal(shape).astype(np.float32)


class LLMDDetector(ThresholdMixin):
    """Full covariance per fixed token position, float64 CPU implementation.

    Inputs: fit [N,H,F], partial_fit [B,H,F], score [H,F]. Torch tensors
    are accepted but copied to CPU; this is not a GPU scoring backend.
    Features must be extracted by the caller under identical conditions.
    """

    def __init__(
        self,
        regularization: float = 1e-6,
        threshold: float | None = None,
        *,
        fixed_noise=None,
        metadata: dict | None = None,
    ):
        self.regularization = scalar(regularization, "regularization", strict=True)
        self.threshold = threshold
        self.fixed_noise = None if fixed_noise is None else array(fixed_noise, "fixed_noise", 2).copy()
        self.metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        self._clear_statistics()

    def _clear_statistics(self) -> None:
        self.n_samples = 0
        self._mean = None
        self._m2 = None
        self._cholesky = None

    @property
    def fitted(self) -> bool:
        return self._cholesky is not None

    @property
    def feature_shape(self) -> tuple[int, int] | None:
        return None if self._mean is None else self._mean.shape

    @property
    def mean(self) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("No feature statistics have been collected")
        return self._mean.copy()

    def fit(self, tokens) -> LLMDDetector:
        """Replace statistics; use partial_fit + finalize for streaming data.

        Refitting clears a previously calibrated threshold, since its scale
        no longer corresponds to these statistics.
        """
        features = array(tokens, "tokens", 3)
        self._clear_statistics()
        self.threshold = None
        self.partial_fit(features)
        return self.finalize()

    def partial_fit(self, tokens) -> LLMDDetector:
        """Merge a batch using centered moments, without retaining features.

        Call only before finalize. Start a new detector to stream a refit.
        """
        if self.fitted:
            raise RuntimeError("Statistics are finalized; start a new detector or call fit to refit")
        features = array(tokens, "tokens", 3)
        if self._mean is not None and features.shape[1:] != self._mean.shape:
            raise ValueError(f"Expected [B,{self._mean.shape[0]},{self._mean.shape[1]}] tokens")
        batch_count = features.shape[0]
        batch_mean = features.mean(axis=0)
        centered = features - batch_mean
        batch_m2 = np.einsum("nhf,nhg->hfg", centered, centered, optimize=True)
        if not np.isfinite(batch_m2).all() or not np.isfinite(batch_mean).all():
            raise ValueError("Feature moments overflowed; check feature scale")
        if self.n_samples == 0:
            new_mean, new_m2 = batch_mean, batch_m2
        else:
            count = self.n_samples + batch_count
            delta = batch_mean - self._mean
            new_mean = self._mean + delta * (batch_count / count)
            new_m2 = (
                self._m2 + batch_m2 + np.einsum("hf,hg->hfg", delta, delta) * (self.n_samples * batch_count / count)
            )
        if not np.isfinite(new_mean).all() or not np.isfinite(new_m2).all():
            raise ValueError("Accumulated moments overflowed; check feature scale")
        self._mean, self._m2 = new_mean, new_m2
        self.n_samples += batch_count
        self.threshold = None
        return self

    def finalize(self) -> LLMDDetector:
        """Use population covariance M2/N and exactly the specified lambda I."""
        if self.fitted:
            return self
        if self.n_samples < 2:
            raise ValueError("At least two observations are required to fit LLMD")
        covariance = self._m2 / self.n_samples
        covariance = (covariance + covariance.swapaxes(-1, -2)) * 0.5
        diagonal = np.arange(covariance.shape[-1])
        covariance[:, diagonal, diagonal] += self.regularization
        try:
            factor = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Covariance is not numerically positive definite; check scale or regularization") from exc
        self._cholesky = factor
        self._m2 = None  # No longer needed online: retain mean and triangular factors only.
        return self

    def score(self, tokens) -> LLMDResult:
        if not self.fitted:
            raise RuntimeError("LLMD must be fitted and finalized before scoring")
        features = array(tokens, "tokens", 2)
        if features.shape != self._mean.shape:
            raise ValueError(f"Expected tokens shape {self._mean.shape}, got {features.shape}")
        delta = features - self._mean
        distances = np.empty(features.shape[0], dtype=np.float64)
        for token in range(features.shape[0]):
            whitened = solve_triangular(self._cholesky[token], delta[token], lower=True, check_finite=False)
            distances[token] = whitened @ whitened
        if not np.isfinite(distances).all():
            raise ValueError("LLMD score overflowed; check feature scale")
        worst = int(np.argmax(distances))
        score = float(distances[worst])
        return LLMDResult(
            score,
            distances.tolist(),
            worst,
            self.threshold,
            None if self.threshold is None else score >= self.threshold,
        )

    def score_batch(self, tokens) -> list[LLMDResult]:
        """Independent [N,H,F] observations, useful for calibration."""
        return [self.score(features) for features in array(tokens, "tokens", 3)]

    def save(self, path: str | Path) -> None:
        """Save statistics, noise, metadata and threshold; never uses pickle."""
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted detector")
        metadata = json.dumps(
            {
                "format_version": 1,
                "regularization": self.regularization,
                "threshold": self.threshold,
                "n_samples": self.n_samples,
                "feature_metadata": self.metadata,
            },
            allow_nan=False,
        )
        with Path(path).open("wb") as handle:
            np.savez_compressed(
                handle,
                mean=self._mean,
                cholesky=self._cholesky,
                fixed_noise=np.empty((0, 0)) if self.fixed_noise is None else self.fixed_noise,
                metadata=np.array(metadata),
            )

    @classmethod
    def load(cls, path: str | Path, *, expected_metadata: dict | None = None) -> LLMDDetector:
        """Optionally reject artifacts from a different checkpoint/layer/task."""
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata"].item()))
            if metadata["format_version"] != 1:
                raise ValueError("Unsupported LLMD artifact version")
            mean = array(artifact["mean"], "saved mean", 2).copy()
            factor = array(artifact["cholesky"], "saved cholesky", 3).copy()
            noise = artifact["fixed_noise"].copy()
        if factor.shape != (mean.shape[0], mean.shape[1], mean.shape[1]):
            raise ValueError("Corrupt LLMD artifact: incompatible mean and Cholesky shapes")
        if np.any(np.diagonal(factor, axis1=1, axis2=2) <= 0) or np.any(np.triu(factor, 1) != 0):
            raise ValueError("Corrupt LLMD artifact: invalid Cholesky factor")
        result = cls(
            metadata["regularization"],
            metadata["threshold"],
            fixed_noise=None if noise.size == 0 else noise,
            metadata=metadata["feature_metadata"],
        )
        for key, value in (expected_metadata or {}).items():
            if key not in result.metadata or result.metadata[key] != value:
                raise ValueError(f"Feature metadata mismatch for {key!r}")
        result.n_samples = integer(metadata["n_samples"], "saved n_samples", 2)
        result._mean, result._cholesky = mean, factor  # noqa: SLF001 - class-owned deserialization
        return result
