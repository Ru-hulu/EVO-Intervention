"""Plan the next action chunk while the current one executes.

Deployment loop (real robot, planning overlaps execution):

    t:    observe o_t, start executing chunk c_t
          in parallel: z_t = E(o_t); predict the latent at the end of c_t with the world model;
          sample K candidates for c_{t+1} from the VLA; refine them with CEM, scoring each by
          rolling it out from the predicted end-of-c_t latent. The VLA only sees o_t, so its
          candidates for c_{t+1} are a proposal distribution, not a policy for o_{t+1}.
    t+1:  c_t finished -> execute the selected c_{t+1}; repeat

The planner works on one environment at a time (batch 1). Candidates and CEM live in the
VLA's normalized action space; `action_transform` maps them to the world model's action units
before `LatentStepContract` aggregates control actions into latent steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from evo_intervention.model.modules.world_model.action_contract import LatentStepContract, StateTransition
from evo_intervention.model.modules.world_model.vjepa2_world_model import VJEPA2ACWorldModel
from evo_intervention.planning.cem import CEMConfig, CEMResult, cross_entropy_method
from evo_intervention.planning.scorers import LatentScorer


@dataclass
class LatentContext:
    """Observed history for one environment: latents [1, T, N, D], states [1, T, S], actions [1, T-1, A_latent]."""

    latents: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor

    def extend(self, latents, states, actions) -> "LatentContext":
        return LatentContext(
            torch.cat((self.latents, latents), dim=1),
            torch.cat((self.states, states), dim=1),
            torch.cat((self.actions, actions), dim=1),
        )

    def last(self, steps: Optional[int]) -> "LatentContext":
        if steps is None or self.latents.shape[1] <= steps:
            return self
        num_actions = self.actions.shape[1] - (steps - 1)
        return LatentContext(self.latents[:, -steps:], self.states[:, -steps:], self.actions[:, num_actions:])


@dataclass
class ChunkPlan:
    chunk: torch.Tensor  # [H_action, A] selected chunk, VLA-normalized space
    cem: CEMResult
    lookahead_latents: Optional[torch.Tensor]  # [1, H_exec, N, D] predicted while the current chunk executes


class LatentChunkPlanner:
    def __init__(
        self,
        world_model: VJEPA2ACWorldModel,
        contract: LatentStepContract,
        state_transition: StateTransition,
        scorer: LatentScorer,
        cem: CEMConfig,
        action_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        max_context_steps: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
    ):
        """max_context_steps: optional extra cap on the history fed to the predictor. The history
        is always trimmed so that history + rollout fits the predictor's causal-mask capacity."""
        self.world_model = world_model
        self.contract = contract
        self.state_transition = state_transition
        self.scorer = scorer
        self.cem = cem
        self.action_transform = action_transform or (lambda a: a)
        self.max_context_steps = max_context_steps
        self.eval_batch_size = eval_batch_size

    def _fit_context(self, context: LatentContext, horizon: int) -> LatentContext:
        """Keep the most recent history that leaves room for a `horizon`-step rollout."""
        limit = self.world_model.max_steps - horizon + 1
        if limit < 1:
            raise ValueError(f"{horizon}-step rollout exceeds the predictor capacity of {self.world_model.max_steps}")
        if self.max_context_steps is not None:
            limit = min(limit, self.max_context_steps)
        return context.last(limit)

    def _actions(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunks [N, H_action, A] -> latent actions [N, H_latent, A_latent], raw actions [N, H_latent, k, A]."""
        actions = self.action_transform(chunks)
        return self.contract.to_latent_actions(actions), self.contract.group(actions)

    @torch.no_grad()
    def observe(
        self, frames: torch.Tensor, states: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> LatentContext:
        """Encode observed frames [1, T, 3, H, W] (uint8 or [0,1]) with states [1, T, S] and executed
        latent actions [1, T-1, A_latent] into a context."""
        latents = self.world_model.encode(self.world_model.preprocess(frames))
        if actions is None:
            actions = states.new_zeros(1, 0, self.contract.latent_action_dim)
        return LatentContext(latents, states, actions)

    @torch.no_grad()
    def lookahead(self, context: LatentContext, executing_chunk: torch.Tensor):
        """Predict through the chunk being executed: returns the extended context and its predicted latents."""
        latent_actions, control_actions = self._actions(executing_chunk[None])
        ctx = self._fit_context(context, latent_actions.shape[1])
        latents, states = self.world_model.rollout(
            ctx.latents,
            ctx.states,
            ctx.actions,
            latent_actions,
            state_transition=self.state_transition,
            transition_actions=control_actions,
        )
        return context.extend(latents, states, latent_actions), latents

    @torch.no_grad()
    def evaluate(self, context: LatentContext, candidates: torch.Tensor) -> torch.Tensor:
        """Score candidate chunks [N, H_action, A] rolled out from the end of `context`."""
        latent_actions, control_actions = self._actions(candidates)
        ctx = self._fit_context(context, latent_actions.shape[1])
        scores = []
        batch = self.eval_batch_size or len(candidates)
        for i in range(0, len(candidates), batch):
            a = latent_actions[i : i + batch]
            n = len(a)
            predicted, states = self.world_model.rollout(
                ctx.latents.expand(n, -1, -1, -1),
                ctx.states.expand(n, -1, -1),
                ctx.actions.expand(n, -1, -1),
                a,
                state_transition=self.state_transition,
                transition_actions=control_actions[i : i + batch],
            )
            scores.append(self.scorer(predicted, states, candidates[i : i + batch]).float())
        return torch.cat(scores)

    @torch.no_grad()
    def plan(
        self,
        context: LatentContext,
        candidates: torch.Tensor,
        executing_chunk: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        postprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> ChunkPlan:
        """candidates [K, H_action, A] for the next chunk; executing_chunk [H_exec, A] or None to plan from now."""
        lookahead_latents = None
        if executing_chunk is not None:
            context, lookahead_latents = self.lookahead(context, executing_chunk)
        result = cross_entropy_method(
            candidates, lambda c: self.evaluate(context, c), self.cem, generator=generator, postprocess=postprocess
        )
        return ChunkPlan(chunk=result.best_candidate, cem=result, lookahead_latents=lookahead_latents)
