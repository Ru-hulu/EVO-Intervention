"""Plain, JSON-serializable outputs. None means undecidable, never False."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass


class JsonResult:
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LLMDResult(JsonResult):
    score: float
    per_token_scores: list[float]
    worst_token: int  # Zero-based index.
    threshold: float | None
    triggered: bool | None


@dataclass(frozen=True)
class ACCResult(JsonResult):
    ready: bool  # False only when a previous chunk is not yet available.
    score: float | None
    raw_score: float | None
    per_axis_mae: list[float] | None
    per_axis_scale: list[float] | None
    per_axis_normalized_error: list[float] | None
    overlap: int
    threshold: float | None
    triggered: bool | None


@dataclass(frozen=True)
class InterventionResult(JsonResult):
    need_intervention: bool | None
    decision_complete: bool
    reasons: list[str]
    unavailable: list[str]
    llmd: LLMDResult | None
    acc: ACCResult | None
