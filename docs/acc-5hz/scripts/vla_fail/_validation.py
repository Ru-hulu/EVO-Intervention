"""Shared validation; tensor inputs are explicitly detached from autograd."""

from __future__ import annotations

import numbers

import numpy as np


def array(value, name: str, ndim: int | tuple[int, ...]) -> np.ndarray:
    # Optional PyTorch interop, including CUDA and bfloat16, without importing torch.
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().double().cpu().numpy()
    try:
        original = np.asarray(value)
        if original.dtype.kind not in "fiu":
            raise ValueError(f"{name} must contain real numbers")
        result = np.asarray(original, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numeric array") from exc
    dimensions = (ndim,) if isinstance(ndim, int) else ndim
    if result.ndim not in dimensions or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must have nonempty dimensions and ndim in {dimensions}; got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return result


def scalar(value, name: str, *, minimum: float = 0.0, strict: bool = False) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or (result <= minimum if strict else result < minimum):
        comparison = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {comparison} {minimum}")
    return result


def integer(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class ThresholdMixin:
    @property
    def threshold(self) -> float | None:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float | None) -> None:
        self._threshold = None if value is None else scalar(value, "threshold")
