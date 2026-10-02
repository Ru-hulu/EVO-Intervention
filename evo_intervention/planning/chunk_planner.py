"""Plan the next action chunk while the current one executes.

Deployment loop (real robot, planning overlaps execution):

    t:    observe o_t, start executing chunk c_t
          in parallel: z_t = E(o_t); predict the latent at the end of c_t with the world model;
          sample K candidates for c_{t+1} from the VLA; refine them with CEM, scoring each by
          rolling it out from the predicted end-of-c_t latent. The VLA only sees o_t, so its
          candidates for c_{t+1} are a proposal distribution, not a policy for o_{t+1}.
    t+1:  c_t finished -> execute the selected c_{t+1}; repeat

As in upstream V-JEPA 2-AC planning, the predictor sees `context_steps` observed latents
(default 1: only the current frame); every later input is its own prediction, which is what
the rollout loss trains (it also starts from one real latent).

`horizon_steps` caps the latent steps rolled out per plan (lookahead + candidate) at the length
the rollout loss was trained on (`loss.rollout_steps`). Pass only the *remaining* part of the
executing chunk; whatever the lookahead uses is taken from the candidate horizon, and candidates
are cut to the steps that can still be scored. The returned chunk is that scored prefix only;
execute it and re-plan, never an unscored VLA tail.

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
    chunk: torch.Tensor  # [H_action, A] selected, scored chunk prefix, VLA-normalized space; execute only this
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
        horizon_steps: int,
        action_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context_steps: int = 1,
        eval_batch_size: Optional[int] = None,
    ):
        """horizon_steps: max latent steps rolled out per plan; set it to the training `loss.rollout_steps`.
        context_steps: observed latents kept before rolling out; 1 matches upstream V-JEPA 2-AC.
        Only observed history is trimmed, never predicted latents."""
        if context_steps < 1:
            raise ValueError("context_steps must be >= 1")
        if horizon_steps < 1:
            raise ValueError("horizon_steps must be >= 1")
        if context_steps + horizon_steps - 1 > world_model.max_steps:
            raise ValueError(
                f"context_steps + horizon_steps - 1 = {context_steps + horizon_steps - 1} exceeds the predictor "
                f"capacity of {world_model.max_steps} latent steps"
            )
        self.world_model = world_model
        self.contract = contract
        self.state_transition = state_transition
        self.scorer = scorer
        self.cem = cem
        self.action_transform = action_transform or (lambda a: a)
        self.horizon_steps = horizon_steps
        self.context_steps = context_steps
        self.eval_batch_size = eval_batch_size

    def _actions(self, chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunks [N, H_action, A] -> latent actions [N, H_latent, A_latent], raw actions [N, H_latent, k, A]."""
        actions = self.action_transform(chunks)
        return self.contract.to_latent_actions(actions), self.contract.group(actions)

    @torch.no_grad()
    def observe(
        self, frames: torch.Tensor, states: torch.Tensor, actions: Optional[torch.Tensor] = None
    ) -> LatentContext:
        """Encode frames [1,T,3,H,W] or [1,T,V,3,H,W] in the training camera order,
        with states [1,T,S] and executed latent actions [1,T-1,A_latent]."""
        latents = self.world_model.encode(self.world_model.preprocess(frames))
        if actions is None:
            actions = states.new_zeros(1, 0, self.contract.latent_action_dim)
        return LatentContext(latents, states, actions)

    @torch.no_grad()
    def lookahead(self, context: LatentContext, executing_chunk: torch.Tensor):
        """Predict through the chunk being executed from an observed context: returns the context
        extended by the predicted latents, and those latents."""
        context = context.last(self.context_steps)
        latent_actions, control_actions = self._actions(executing_chunk[None])
        latents, states = self.world_model.rollout(
            context.latents,
            context.states,
            context.actions,
            latent_actions,
            state_transition=self.state_transition,
            transition_actions=control_actions,
        )
        return context.extend(latents, states, latent_actions), latents

    @torch.no_grad()
    def evaluate(self, context: LatentContext, candidates: torch.Tensor) -> torch.Tensor:
        """Score candidate chunks [N, H_action, A] rolled out from the end of `context`, as returned
        by `plan`'s trimming or by `lookahead`; it is used as given."""
        latent_actions, control_actions = self._actions(candidates)
        scores = []
        batch = self.eval_batch_size or len(candidates)
        for i in range(0, len(candidates), batch):
            a = latent_actions[i : i + batch]
            n = len(a)
            predicted, states = self.world_model.rollout(
                context.latents.expand(n, -1, -1, -1),
                context.states.expand(n, -1, -1),
                context.actions.expand(n, -1, -1),
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
        """candidates [K, H_action, A] for the next chunk; executing_chunk [H_exec, A]: the not yet executed
        rest of the current chunk, or None to plan from now. Candidates are cut to the horizon left after
        the lookahead; `ChunkPlan.chunk` has that (possibly shorter) length."""
        lookahead_steps = 0 if executing_chunk is None else self.contract.num_latent_steps(len(executing_chunk))
        candidate_steps = self.horizon_steps - lookahead_steps
        if candidate_steps < 1:
            raise ValueError(
                f"Lookahead of {lookahead_steps} latent steps leaves no room in horizon_steps={self.horizon_steps}; "
                "plan later in the executing chunk"
            )
        candidates = candidates[:, : candidate_steps * self.contract.actions_per_latent_step]

        context = context.last(self.context_steps)
        lookahead_latents = None
        if executing_chunk is not None:
            context, lookahead_latents = self.lookahead(context, executing_chunk)
        result = cross_entropy_method(
            candidates, lambda c: self.evaluate(context, c), self.cem, generator=generator, postprocess=postprocess
        )
        return ChunkPlan(chunk=result.best_candidate, cem=result, lookahead_latents=lookahead_latents)
