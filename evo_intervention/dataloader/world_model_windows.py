"""Time windows of continuous robot trajectories for world-model training.

Unlike the VLA samples (one frame per camera), a window keeps the time axis: frames sampled
every `contract.actions_per_latent_step` control steps, the recorded state at each sampled
frame, and the control actions between consecutive frames aggregated into latent actions.

The training dataset is not fixed yet, so storage is behind `EpisodeSource`. `NpzEpisodeSource`
is a minimal reference format; a LeRobot/real-robot source should implement the same methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from evo_intervention.model.modules.world_model.action_contract import LatentStepContract


class EpisodeSource(Protocol):
    def __len__(self) -> int: ...

    def episode_length(self, episode: int) -> int:
        """Number of control steps (frames, states and actions are aligned per control step)."""

    def load_frames(self, episode: int, indices: np.ndarray) -> np.ndarray:
        """uint8 frames [len(indices), H, W, 3] from the single camera used by the world model."""

    def load_lowdim(self, episode: int, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """states [stop-start, S] and actions [stop-start, A] for control steps [start, stop)."""


class NpzEpisodeSource:
    """One `.npz` per episode with `frames` [L, H, W, 3] uint8, `states` [L, S], `actions` [L, A]."""

    def __init__(self, root: str | Path, camera_key: str = "frames"):
        self.files = sorted(Path(root).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz episodes in {root}")
        self.camera_key = camera_key
        self._lengths = [len(np.load(f, mmap_mode="r")["actions"]) for f in self.files]

    def __len__(self):
        return len(self.files)

    def episode_length(self, episode):
        return self._lengths[episode]

    def load_frames(self, episode, indices):
        return np.load(self.files[episode])[self.camera_key][indices]

    def load_lowdim(self, episode, start, stop):
        data = np.load(self.files[episode])
        return data["states"][start:stop], data["actions"][start:stop]


def split_episodes(num_episodes: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split by episode so overlapping windows never cross train/val."""
    order = np.random.default_rng(seed).permutation(num_episodes).tolist()
    num_val = round(num_episodes * val_fraction)
    if 0 < val_fraction and num_val == 0 and num_episodes > 1:
        num_val = 1
    return sorted(order[num_val:]), sorted(order[:num_val])


class TrajectoryWindowDataset(Dataset):
    """Windows of `window_frames` latent frames.

    Windows shorter than `window_frames` but with at least `min_frames` frames are padded by
    repeating the last frame and marked invalid in `valid`; set `min_frames = window_frames`
    to keep only full windows.
    """

    def __init__(
        self,
        source: EpisodeSource,
        contract: LatentStepContract,
        window_frames: int,
        min_frames: int | None = None,
        episodes: Sequence[int] | None = None,
        window_stride: int | None = None,
    ):
        self.source = source
        self.contract = contract
        self.window_frames = window_frames
        self.min_frames = window_frames if min_frames is None else min_frames
        if not 2 <= self.min_frames <= window_frames:
            raise ValueError("Need 2 <= min_frames <= window_frames")
        step = contract.actions_per_latent_step
        window_stride = window_stride or step
        self.index = []
        for ep in range(len(source)) if episodes is None else episodes:
            # A window with F frames spans (F - 1) * step control actions after its first frame.
            last_start = source.episode_length(ep) - (self.min_frames - 1) * step - 1
            self.index.extend((ep, start) for start in range(0, last_start + 1, window_stride))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        episode, start = self.index[i]
        step, F = self.contract.actions_per_latent_step, self.window_frames
        length = self.source.episode_length(episode)
        num_frames = min(F, (length - 1 - start) // step + 1)

        frame_idx = start + step * np.arange(num_frames)
        frames = self.source.load_frames(episode, frame_idx)
        states, actions = self.source.load_lowdim(episode, start, start + (num_frames - 1) * step + 1)
        states = states[::step]
        raw = torch.as_tensor(actions[: (num_frames - 1) * step], dtype=torch.float32)
        actions, control_actions = self.contract.to_latent_actions(raw), self.contract.group(raw)

        frames = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)  # [F, 3, H, W]
        states = torch.as_tensor(states, dtype=torch.float32)
        valid = torch.zeros(F, dtype=torch.bool)
        valid[:num_frames] = True
        if num_frames < F:
            pad = F - num_frames
            frames = torch.cat((frames, frames[-1:].expand(pad, -1, -1, -1)))
            states = torch.cat((states, states[-1:].expand(pad, -1)))
            actions = torch.cat((actions, actions.new_zeros(pad, actions.shape[-1])))
            control_actions = torch.cat((control_actions, control_actions.new_zeros(pad, *control_actions.shape[1:])))
        # control_actions [F-1, k, A]: raw actions per latent step, for deployment-like state updates.
        return {
            "frames": frames,
            "states": states,
            "actions": actions,
            "control_actions": control_actions,
            "valid": valid,
        }
