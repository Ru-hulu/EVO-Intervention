"""Paper OR fusion, with explicit missing-history / missing-threshold states."""

from __future__ import annotations

from .acc import ACCDetector
from .llmd import LLMDDetector
from .results import ACCResult
from .results import InterventionResult
from .results import LLMDResult


def combine_results(*, llmd: LLMDResult | None = None, acc: ACCResult | None = None) -> InterventionResult:
    """A missing argument disables that signal; a present unknown stays unknown.

    True OR unknown -> True; False OR unknown -> None; False OR False -> False.
    decision_complete reports whether all enabled detectors had a decision.
    """
    if llmd is None and acc is None:
        raise ValueError("At least one detector result is required")
    reasons, unavailable = [], []
    if llmd is not None:
        if llmd.triggered:
            reasons.append("llmd_feature_ood")
        elif llmd.triggered is None:
            unavailable.append("llmd_threshold_missing")
    if acc is not None:
        if acc.triggered:
            reasons.append("acc_action_inconsistency")
        elif not acc.ready:
            unavailable.append("acc_history_missing")
        elif acc.triggered is None:
            unavailable.append("acc_threshold_missing")
    decision = True if reasons else (None if unavailable else False)
    return InterventionResult(decision, not unavailable, reasons, unavailable, llmd, acc)


class VLAFAILDetector:
    """Python integration API. Use LLMD only by leaving acc=None.

    This returns a takeover recommendation; it does not call an LLM, generate
    actions, or implement the independent semantic intervention layer.
    """

    def __init__(self, *, llmd: LLMDDetector | None = None, acc: ACCDetector | None = None):
        if llmd is None and acc is None:
            raise ValueError("Enable at least one of LLMD and ACC")
        self.llmd, self.acc = llmd, acc

    def step(self, *, tokens=None, positions=None, executed_steps: int | None = None) -> InterventionResult:
        if self.llmd is not None and tokens is None:
            raise ValueError("tokens are required because LLMD is enabled")
        if self.acc is not None and positions is None:
            raise ValueError("positions are required because ACC is enabled")
        if self.llmd is None and tokens is not None:
            raise ValueError("tokens provided but LLMD is disabled")
        if self.acc is None and (positions is not None or executed_steps is not None):
            raise ValueError("action inputs provided but ACC is disabled")
        # Score LLMD before mutating ACC state, so feature errors leave history intact.
        llmd_result = None if self.llmd is None else self.llmd.score(tokens)
        acc_result = None if self.acc is None else self.acc.update(positions, executed_steps)
        return combine_results(llmd=llmd_result, acc=acc_result)

    def reset(self) -> None:
        """Clear per-episode action history while retaining LLMD statistics."""
        if self.acc is not None:
            self.acc.reset()
