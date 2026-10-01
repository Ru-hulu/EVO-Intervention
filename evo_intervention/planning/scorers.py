"""Candidate scorers over predicted JEPA latents. Higher is better.

The progress indicator is developed separately; `ProgressScorer` wraps any callable that maps
latents to a progress value. It is evaluated on *predicted* latents, so it should be validated
on predictor outputs, not only on encoder outputs of real frames.
"""

from __future__ import annotations

from typing import Callable, Literal, Protocol, Sequence

import torch

from evo_intervention.model.modules.world_model.vjepa2_world_model import latent_l1


class LatentScorer(Protocol):
    def __call__(
        self, predicted_latents: torch.Tensor, predicted_states: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """predicted_latents [N, H, P, D], predicted_states [N, H', S], candidates [N, H_action, A] -> [N]."""


class GoalLatentScorer:
    """Negative latent L1 to a goal latent, e.g. the encoded frame k actions ahead of a reference.

    `step` selects which predicted latent is compared (-1 = end of the candidate chunk).
    """

    def __init__(self, goal_latent: torch.Tensor, step: int = -1, loss_exp: float = 1.0):
        self.goal_latent = goal_latent.reshape(1, *goal_latent.shape[-2:])
        self.step = step
        self.loss_exp = loss_exp

    def __call__(self, predicted_latents, predicted_states, candidates):
        goal = self.goal_latent.to(predicted_latents)
        return -latent_l1(predicted_latents[:, self.step], goal, self.loss_exp)


class ProgressScorer:
    """Wrap a latent progress model `progress_fn(latents [M, P, D]) -> [M]`."""

    def __init__(
        self, progress_fn: Callable[[torch.Tensor], torch.Tensor], reduce: Literal["final", "mean", "max"] = "final"
    ):
        self.progress_fn = progress_fn
        self.reduce = reduce

    def __call__(self, predicted_latents, predicted_states, candidates):
        N, H = predicted_latents.shape[:2]
        progress = self.progress_fn(predicted_latents.flatten(0, 1)).view(N, H)
        if self.reduce == "final":
            return progress[:, -1]
        if self.reduce == "mean":
            return progress.mean(1)
        return progress.amax(1)


class WeightedScorer:
    def __init__(self, terms: Sequence[tuple[float, LatentScorer]]):
        self.terms = list(terms)

    def __call__(self, predicted_latents, predicted_states, candidates):
        return sum(w * s(predicted_latents, predicted_states, candidates) for w, s in self.terms)
