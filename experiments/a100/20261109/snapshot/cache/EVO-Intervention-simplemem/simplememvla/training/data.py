"""Full-history video samples; supervision is never inserted into the prompt."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers.video_utils import VideoMetadata

from qwen_training.progress.data import read_video


def sample_indices(total, source_fps, fps, max_frames):
    if total < 1 or source_fps <= 0 or fps <= 0 or max_frames < 2 or max_frames % 2:
        raise ValueError("Require positive fps/length and an even max_frames >= 2")
    indices = np.unique(np.r_[np.arange(0, total, max(1, source_fps / fps)).astype(int), total - 1])
    if len(indices) > max_frames:
        indices = indices[np.linspace(0, len(indices) - 1, max_frames).astype(int)]
    if len(indices) % 2:
        indices = np.r_[indices, indices[-1]]
    return indices.tolist()


def progress_at(label, indices):
    if label is None:
        return [-100.0] * len(indices)
    if "csv" in label:
        path = Path(label["csv"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != label["sha256"]:
            raise ValueError(f"Progress changed since import: {path}")
        values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        positions = np.searchsorted(values[:, 0], indices)
        if np.any(positions >= len(values)) or not np.array_equal(values[positions, 0], indices):
            raise ValueError(f"Progress missing requested source frames: {path}")
        return values[positions, 2].tolist()
    if "dense" in label:
        return [label["dense"][i] for i in indices]
    ends = np.asarray(label["ends"])
    result = []
    for index in indices:
        done = int(np.searchsorted(ends, index, side="right"))
        start = ends[done - 1] + 1 if done else 0
        fraction = max(0, (index - start) / max(1, ends[done] - start + 1)) if done < len(ends) else 0
        result.append(min(1.0, (done + fraction) / len(ends)))
    return result


class RewardDataset(Dataset):
    def __init__(
        self,
        records,
        media_root,
        split,
        fps=2.0,
        max_frames=120,
        width=320,
        height=256,
        views="all",
        allow_derived=False,
        pairs=None,
    ):
        self.root = Path(media_root)
        self.fps, self.max_frames = fps, max_frames
        self.width, self.height, self.views = width, height, views
        if width % 32 or height % 32 or min(width, height) <= 0:
            raise ValueError("Per-view width/height must be positive multiples of 32")
        all_records = [json.loads(x) for x in Path(records).read_text().splitlines() if x.strip()]
        self.records = [x for x in all_records if x["split"] == split]
        if not self.records:
            raise ValueError(f"No records in split {split}")
        if not allow_derived and any(
            x["progress"] and x["progress"]["kind"].startswith("derived_") for x in self.records
        ):
            raise ValueError("Event-normalized progress is a proxy: explicitly enable allow_derived")
        self.by_id = {x["id"]: x for x in self.records}
        self.pairs = None
        if pairs:
            self.pairs = [json.loads(x) for x in Path(pairs).read_text().splitlines() if x.strip()]
            self.pairs = [x for x in self.pairs if x["split"] == split]
            for pair in self.pairs:
                a, b = self.by_id[pair["a"]], self.by_id[pair["b"]]
                if a["id"] == b["id"] or a["task"] != b["task"] or a["instruction"] != b["instruction"]:
                    raise ValueError("Preference requires distinct same-task, same-instruction trajectories")
                if pair["label"] not in (0, 1):
                    raise ValueError("Preference label must be 0 or 1 (1 = A preferred)")
            if not self.pairs:
                raise ValueError(f"No preference pairs in {split}")

    def __len__(self):
        return len(self.pairs) if self.pairs is not None else len(self.records)

    def feature(self, row):
        start, stop = row.get("frame_start", 0), row.get("frame_stop", row["total_frames"])
        indices = [start + i for i in sample_indices(stop - start, row["fps"], self.fps, self.max_frames)]
        paths = row["videos"][:1] if self.views == "main" else row["videos"]
        sizes = row["video_sizes"][: len(paths)]
        cameras = []
        for file, size in zip(paths, sizes, strict=True):
            path = self.root / file
            if not path.is_file() or (size is not None and path.stat().st_size != size):
                raise ValueError(f"Media missing or differs from audited snapshot: {path}")
            frames = read_video(path, indices)
            cameras.append(
                np.stack(
                    [
                        np.asarray(Image.fromarray(frame).resize((self.width, self.height), Image.Resampling.BILINEAR))
                        for frame in frames
                    ]
                )
            )
        video = np.concatenate(cameras, axis=2)
        success = [-100.0] * (len(indices) // 2)
        # Episode success is only an endpoint target, never broadcast to unfinished prefixes.
        if row["success"] is not None:
            success[-1] = float(row["success"])
        return dict(
            instruction=row["instruction"],
            videos=[video],
            images=[],
            video_metadata=[
                VideoMetadata(
                    total_num_frames=row["total_frames"],
                    fps=row["fps"],
                    frames_indices=indices,
                    duration=row["total_frames"] / row["fps"],
                )
            ],
            progress_targets=progress_at(row["progress"], indices[1::2]),
            success_targets=success,
        )

    def __getitem__(self, index):
        if self.pairs is None:
            return self.feature(self.records[index])
        pair = self.pairs[index]
        a, b = self.feature(self.by_id[pair["a"]]), self.feature(self.by_id[pair["b"]])
        for key in ("videos", "video_metadata", "progress_targets", "success_targets"):
            a[key] += b[key]
        a["preference_targets"] = float(pair["label"])
        return a
