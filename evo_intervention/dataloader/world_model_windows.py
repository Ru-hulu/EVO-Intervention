"""Time windows of continuous robot trajectories for world-model training.

Unlike the VLA samples (one frame per camera), a window keeps the time axis: frames sampled
every `contract.actions_per_latent_step` control steps, the recorded state at each sampled
frame, and the control actions between consecutive frames aggregated into latent actions.
Multiple cameras retain a separate view axis in the configured order.

`NpzEpisodeSource` reads reference arrays; `LeRobotV3EpisodeSource` reads native robot recordings.
"""

from __future__ import annotations

import json
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
        """uint8 frames [F,H,W,3] or [F,V,H,W,3] for the configured ordered cameras."""

    def load_lowdim(self, episode: int, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """states [stop-start, S] and actions [stop-start, A] for control steps [start, stop)."""


class NpzEpisodeSource:
    """One `.npz` per episode: uint8 frames [L,H,W,3] or [L,V,H,W,3], states [L,S], actions [L,A]."""

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


class LeRobotV3EpisodeSource:
    """Native v3 parquet/video episodes; state and absolute actions normalized to [-1, 1]."""

    def __init__(self, root: str | Path, camera_key: str | None = None, *, camera_keys: Sequence[str] | None = None):
        import pandas as pd

        self.root = Path(root).expanduser()
        self.camera_key = camera_key
        if camera_key is not None and camera_keys is not None:
            raise ValueError("Use camera_key or camera_keys, not both")
        if isinstance(camera_keys, str):
            raise ValueError("camera_keys must be a list of camera names")
        self.multi_camera = camera_keys is not None
        self.camera_keys = list(camera_keys) if self.multi_camera else [camera_key]
        if not self.camera_keys or any(not isinstance(key, str) or not key for key in self.camera_keys):
            raise ValueError("Configure at least one nonempty camera key")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must contain distinct cameras")
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v3.0":
            raise ValueError("LeRobotV3EpisodeSource requires codebase_version=v3.0")
        for key in self.camera_keys:
            if self.info["features"][key]["dtype"] != "video":
                raise ValueError(f"{key} is not a video camera")
        paths = sorted(self.root.glob("meta/episodes/*/*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No episode metadata in {self.root}")
        self.episodes = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        self.episodes = self.episodes.sort_values("episode_index").reset_index(drop=True)
        if self.episodes["episode_index"].duplicated().any():
            raise ValueError("Duplicate episode indices in LeRobot metadata")
        stats = json.loads((self.root / "meta/stats.json").read_text())
        self.lowdim_stats = {
            key: (
                np.asarray(stats[key]["min"], dtype=np.float32).reshape(-1),
                np.asarray(stats[key]["max"], dtype=np.float32).reshape(-1),
            )
            for key in ("observation.state", "action")
        }
        self._cached_path = None
        self._cached_file = None
        self._cached_episode = None
        self._cached_rows = None

    def __len__(self):
        return len(self.episodes)

    def episode_length(self, episode):
        return int(self.episodes.iloc[episode]["length"])

    def _episode_rows(self, episode):
        import pandas as pd

        if episode == self._cached_episode:
            return self._cached_rows
        meta = self.episodes.iloc[episode]
        path = self.root / self.info["data_path"].format(
            chunk_index=int(meta["data/chunk_index"]), file_index=int(meta["data/file_index"])
        )
        if path != self._cached_path:
            self._cached_file = pd.read_parquet(
                path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"]
            )
            self._cached_path = path
        rows = self._cached_file[self._cached_file["episode_index"] == int(meta["episode_index"])]
        rows = rows.sort_values("frame_index").reset_index(drop=True)
        if not np.array_equal(rows["frame_index"].to_numpy(), np.arange(self.episode_length(episode))):
            raise ValueError(f"Missing or misaligned frames in episode {meta['episode_index']}")
        self._cached_episode, self._cached_rows = episode, rows
        return rows

    def load_frames(self, episode, indices):
        from evo_intervention.dataloader.gr00t_lerobot.video import get_frames_by_timestamps

        meta = self.episodes.iloc[episode]
        timestamps = self._episode_rows(episode)["timestamp"].to_numpy()[indices]
        views = []
        for key in self.camera_keys:
            prefix = f"videos/{key}"
            path = self.root / self.info["video_path"].format(
                video_key=key,
                chunk_index=int(meta[f"{prefix}/chunk_index"]),
                file_index=int(meta[f"{prefix}/file_index"]),
            )
            # Each camera can have a different offset inside its own video file.
            camera_timestamps = timestamps + float(meta[f"{prefix}/from_timestamp"])
            views.append(get_frames_by_timestamps(str(path), camera_timestamps, video_backend="pyav"))
        return np.stack(views, axis=1) if self.multi_camera else views[0]

    def load_lowdim(self, episode, start, stop):
        rows = self._episode_rows(episode).iloc[start:stop]
        normalized = []
        for key in ("observation.state", "action"):
            values = np.stack(rows[key]).astype(np.float32)
            minimum, maximum = self.lowdim_stats[key]
            span = maximum - minimum
            values = np.where(span > 1e-6, 2 * (values - minimum) / np.maximum(span, 1e-6) - 1, 0)
            normalized.append(values.astype(np.float32))
        return tuple(normalized)


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

        frames = torch.from_numpy(np.ascontiguousarray(frames)).movedim(-1, -3)  # [F,3,H,W] or [F,V,3,H,W]
        states = torch.as_tensor(states, dtype=torch.float32)
        valid = torch.zeros(F, dtype=torch.bool)
        valid[:num_frames] = True
        if num_frames < F:
            pad = F - num_frames
            frames = torch.cat((frames, frames[-1:].expand(pad, *frames.shape[1:])))
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
