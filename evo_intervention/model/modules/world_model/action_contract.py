"""Action/state contract between VLA control actions and JEPA latent steps.

One JEPA latent step spans `actions_per_latent_step` control actions. The VLA emits chunks of
control actions; the world model consumes one aggregated action per latent step. The dataset
is not fixed yet, so the reduction and the state update are configurable and must be recorded
together with every trained predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch

ActionReduce = Literal["sum", "mean", "last", "concat"]


@dataclass(frozen=True)
class LatentStepContract:
    actions_per_latent_step: int = 1
    action_reduce: ActionReduce = "sum"
    action_dim: int = 7
    state_dim: int = 7

    def __post_init__(self):
        if self.actions_per_latent_step < 1:
            raise ValueError("actions_per_latent_step must be >= 1")
        if self.action_reduce not in ("sum", "mean", "last", "concat"):
            raise ValueError(f"Unknown action_reduce: {self.action_reduce}")

    @property
    def latent_action_dim(self) -> int:
        """Width of one aggregated action, i.e. the predictor's `action_embed_dim`."""
        if self.action_reduce == "concat":
            return self.action_dim * self.actions_per_latent_step
        return self.action_dim

    def num_latent_steps(self, num_actions: int) -> int:
        if num_actions % self.actions_per_latent_step:
            raise ValueError(
                f"{num_actions} control actions do not divide into latent steps of {self.actions_per_latent_step}"
            )
        return num_actions // self.actions_per_latent_step

    def group(self, actions: torch.Tensor) -> torch.Tensor:
        """[..., H_action, A] control actions -> [..., H_latent, k, A] raw actions per latent step.

        This, not the aggregated latent action, is what a `StateTransition` consumes: e.g. summed
        Euler deltas are not the composition of the individual rotations.
        """
        steps = self.num_latent_steps(actions.shape[-2])
        return actions.unflatten(-2, (steps, self.actions_per_latent_step))

    def to_latent_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """[..., H_action, A] control actions -> [..., H_latent, A_latent] predictor actions."""
        grouped = self.group(actions)
        if self.action_reduce == "sum":
            return grouped.sum(dim=-2)
        if self.action_reduce == "mean":
            return grouped.mean(dim=-2)
        if self.action_reduce == "last":
            return grouped[..., -1, :]
        return grouped.flatten(-2)


class StateTransition(Protocol):
    """Deployment-time state update over one latent step; training may use recorded states instead."""

    def __call__(self, state: torch.Tensor, control_actions: torch.Tensor) -> torch.Tensor:
        """state [N, S], control_actions [N, k, A] executed in order -> next state [N, S]."""


class HoldStateTransition:
    """Keep the last observed state. Baseline when no validated state model exists."""

    def __call__(self, state: torch.Tensor, control_actions: torch.Tensor) -> torch.Tensor:
        return state


def euler_xyz_to_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Extrinsic xyz Euler angles (scipy "xyz") [..., 3] -> rotation matrices [..., 3, 3]."""
    x, y, z = angles.unbind(-1)
    cx, cy, cz = x.cos(), y.cos(), z.cos()
    sx, sy, sz = x.sin(), y.sin(), z.sin()
    row0 = torch.stack((cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz), dim=-1)
    row1 = torch.stack((cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz), dim=-1)
    row2 = torch.stack((-sy, sx * cy, cx * cy), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [..., 3, 3] -> extrinsic xyz Euler angles [..., 3]."""
    y = torch.asin((-matrix[..., 2, 0]).clamp(-1.0, 1.0))
    x = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    z = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    return torch.stack((x, y, z), dim=-1)


class DeltaPoseStateTransition:
    """Torch port of upstream `compute_new_pose` for [xyz, euler_xyz, gripper] states.

    Valid only when each control action is an end-effector delta in the same frame as the state,
    as in V-JEPA 2-AC on DROID. Control actions are applied one by one, so rotations compose
    correctly. `gripper_mode="absolute"` copies the action's gripper value.
    """

    def __init__(self, gripper_mode: Literal["delta", "absolute"] = "delta", gripper_range=(0.0, 1.0)):
        self.gripper_mode = gripper_mode
        self.gripper_range = gripper_range

    def __call__(self, state: torch.Tensor, control_actions: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] != 7 or control_actions.shape[-1] != 7 or control_actions.ndim != state.ndim + 1:
            raise ValueError("DeltaPoseStateTransition expects states [N, 7] and control actions [N, k, 7]")
        for action in control_actions.unbind(-2):
            state = self.step(state, action)
        return state

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        xyz = state[..., :3] + action[..., :3]
        rotation = euler_xyz_to_matrix(action[..., 3:6]) @ euler_xyz_to_matrix(state[..., 3:6])
        gripper = action[..., 6:] if self.gripper_mode == "absolute" else state[..., 6:] + action[..., 6:]
        gripper = gripper.clamp(*self.gripper_range)
        return torch.cat((xyz, matrix_to_euler_xyz(rotation), gripper), dim=-1)


def build_state_transition(name: str, **kwargs) -> StateTransition:
    if name == "hold":
        return HoldStateTransition()
    if name == "delta_pose":
        return DeltaPoseStateTransition(**kwargs)
    raise ValueError(f"Unknown state transition: {name}")
