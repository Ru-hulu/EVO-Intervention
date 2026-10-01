"""Cross-entropy method over action chunks, initialized from VLA candidates.

Adapted from the search structure of V-JEPA 2-AC `notebooks/utils/mpc_utils.py::cem`, but:
- the first population is the VLA's own candidates, not a zero-mean Gaussian;
- it optimizes the full action dimension of the chunk;
- the evaluator is pluggable (higher score is better);
- it returns the best *evaluated* candidate, never an unevaluated distribution mean;
- non-finite scores (NaN/inf from a broken rollout or scorer) are never selected;
- sampling uses an explicit torch.Generator and every iteration is traced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

Evaluator = Callable[[torch.Tensor], torch.Tensor]  # [N, H, A] -> scores [N]


@dataclass
class CEMConfig:
    iterations: int = 3  # refinement rounds after scoring the VLA candidates; 0 = pick the best VLA candidate
    population: int = 64  # new samples per refinement round
    num_elites: int = 8
    momentum_mean: float = 0.0  # weight kept on the previous mean
    momentum_std: float = 0.0
    min_std: float = 1e-3
    # Bounds for CEM samples in the candidate space. No default: set them from the action
    # normalization in use (e.g. q99 normalization allows values beyond [-1, 1]).
    # VLA candidates are evaluated as given and never clipped.
    action_low: Optional[float] = None
    action_high: Optional[float] = None
    keep_elites: bool = True  # carry scored elites into the next round without re-evaluating
    evaluate_mean: bool = True  # also score the distribution mean each round


@dataclass
class CEMResult:
    best_candidate: torch.Tensor  # [H, A]
    best_score: float
    num_evaluations: int
    trace: list[dict] = field(default_factory=list)


def _fit(elites: torch.Tensor, prev_mean, prev_std, cfg: CEMConfig):
    mean = elites.mean(0)
    std = elites.std(0, unbiased=False) if len(elites) > 1 else torch.zeros_like(mean)
    if prev_mean is not None:
        mean = cfg.momentum_mean * prev_mean + (1 - cfg.momentum_mean) * mean
        std = cfg.momentum_std * prev_std + (1 - cfg.momentum_std) * std
    return mean, std.clamp_min(cfg.min_std)


def _clip(x: torch.Tensor, cfg: CEMConfig) -> torch.Tensor:
    if cfg.action_low is None and cfg.action_high is None:
        return x
    return x.clamp(cfg.action_low, cfg.action_high)


def _finite(scores: torch.Tensor) -> torch.Tensor:
    """Map NaN/inf scores to -inf so they rank last."""
    return torch.where(torch.isfinite(scores), scores, torch.full_like(scores, float("-inf")))


@torch.no_grad()
def cross_entropy_method(
    initial_candidates: torch.Tensor,
    evaluate: Evaluator,
    cfg: CEMConfig,
    generator: Optional[torch.Generator] = None,
    postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> CEMResult:
    """initial_candidates [K, H, A] (e.g. VLA samples); `postprocess` can snap e.g. gripper dims."""
    if initial_candidates.ndim != 3:
        raise ValueError(f"Expected [K, H, A] candidates, got {tuple(initial_candidates.shape)}")
    post = postprocess or (lambda x: x)

    pool = post(initial_candidates)
    raw_scores = evaluate(pool)
    scores = _finite(raw_scores)
    num_evals = len(pool)
    trace = [{"iteration": 0, "source": "vla", "scores": raw_scores.cpu()}]
    if not torch.isfinite(scores).any():
        raise RuntimeError("Every initial candidate has a non-finite score")
    best_idx = scores.argmax()
    best, best_score = pool[best_idx].clone(), scores[best_idx].item()

    mean = std = None
    for it in range(1, cfg.iterations + 1):
        num_finite = int(torch.isfinite(scores).sum())
        elite_idx = scores.topk(min(cfg.num_elites, num_finite)).indices
        elites, elite_scores = pool[elite_idx], scores[elite_idx]
        mean, std = _fit(elites, mean, std, cfg)

        noise = torch.randn((cfg.population, *mean.shape), generator=generator, device=mean.device, dtype=mean.dtype)
        samples = mean + std * noise
        if cfg.evaluate_mean:
            samples = torch.cat((mean[None], samples), dim=0)
        samples = post(_clip(samples, cfg))
        raw_scores = evaluate(samples)
        sample_scores = _finite(raw_scores)
        num_evals += len(samples)

        if cfg.keep_elites:
            pool, scores = torch.cat((elites, samples)), torch.cat((elite_scores, sample_scores))
        else:
            pool, scores = samples, sample_scores
        round_best = sample_scores.argmax()
        if sample_scores[round_best].item() > best_score:
            best, best_score = samples[round_best].clone(), sample_scores[round_best].item()
        trace.append(
            {"iteration": it, "source": "cem", "mean": mean.cpu(), "std": std.cpu(), "scores": raw_scores.cpu()}
        )

    return CEMResult(best_candidate=best, best_score=best_score, num_evaluations=num_evals, trace=trace)
