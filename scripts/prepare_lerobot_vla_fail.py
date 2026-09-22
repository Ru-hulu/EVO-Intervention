"""Export selected LeRobot v3 episodes as portable, observation-only NPZ bundles.

Uses recorded frame timestamps plus the metadata's per-video offset. Applies
the installed LeRobot pi05's exact resize/pad function; retains float32 pixels.
Never substitutes logged actions for unrecorded action-chunk predictions.
"""

import argparse
import json
from pathlib import Path

import av
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
import numpy as np
import pyarrow.parquet as pq
import torch


def episode_metadata(root, episode):
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        columns = [name for name in pq.read_schema(path).names if not name.startswith("stats/")]
        table = pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode)])
        if len(table):
            if len(table) != 1:
                raise ValueError("Episode metadata must be unique")
            return table.to_pylist()[0]
    raise ValueError(f"Episode {episode} was not found")


def video_frames(path, timestamps, fps):
    result = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        decoder = None
        previous_time = -np.inf
        for target in timestamps:
            if decoder is None or target - previous_time > 0.5:
                container.seek(max(0, int(target / float(stream.time_base))), stream=stream, backward=True)
                decoder = container.decode(stream)
            for frame in decoder:
                current_time = float(frame.pts * stream.time_base)
                if current_time >= target - 0.5 / fps:
                    if abs(current_time - target) > 1.1 / fps:
                        raise ValueError(f"Video timestamp mismatch: {path}, target={target}, decoded={current_time}")
                    array = frame.to_ndarray(format="rgb24")
                    image = torch.from_numpy(array).float().div_(255)[None]
                    image = resize_with_pad_torch(image, 224, 224)[0].numpy()
                    result.append(image)
                    previous_time = current_time
                    break
            else:
                raise ValueError(f"Video ended before timestamp {target}: {path}")
    return np.stack(result)


def prepare(
    root, episode, output, *, stride=10, sample_count=None, base_camera="observation.images.right_environment_1"
):
    root, output = Path(root), Path(output)
    info = json.loads((root / "meta/info.json").read_text())
    meta = episode_metadata(root, episode)
    data_path = root / info["data_path"].format(
        chunk_index=meta["data/chunk_index"], file_index=meta["data/file_index"]
    )
    table = pq.read_table(
        data_path, filters=[("episode_index", "=", episode)], columns=["frame_index", "timestamp", "observation.state"]
    )
    frames = table["frame_index"].to_numpy()
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Expected contiguous, episode-relative frame indices")
    if stride <= 0 or (sample_count is not None and sample_count < 2):
        raise ValueError("stride must be positive and sample_count >= 2")
    indices = (
        np.arange(0, len(frames), stride)
        if sample_count is None
        else np.unique(np.linspace(0, len(frames) - 1, min(sample_count, len(frames)), dtype=int))
    )
    times = table["timestamp"].to_numpy()[indices]
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[indices]
    destination = output / f"episode_{episode:06d}.npz"
    if destination.exists():
        raise FileExistsError(destination)
    output.mkdir(parents=True, exist_ok=True)
    cameras = {
        "base": base_camera,
        "left_wrist": "observation.images.left_wrist",
        "right_wrist": "observation.images.right_wrist",
    }
    arrays = {}
    for name, key in cameras.items():
        prefix = f"videos/{key}/"
        path = root / info["video_path"].format(
            video_key=key, chunk_index=meta[prefix + "chunk_index"], file_index=meta[prefix + "file_index"]
        )
        arrays[f"images_{name}"] = video_frames(path, times + meta[prefix + "from_timestamp"], info["fps"])
        print(f"episode={episode} camera={name} frames={len(indices)}", flush=True)
    labels = {}
    if meta.get("episode_success") in ("success", "failure"):
        labels["success"] = np.array(meta["episode_success"] == "success")
    provenance = {
        "dataset_root": str(root),
        "episode_index": episode,
        "fps": info["fps"],
        "source_frames": len(frames),
        "sample_stride": stride if sample_count is None else None,
        "sample_count": sample_count,
        "camera_map": cameras,
        "label": meta.get("episode_success", "unknown"),
        "collector_policy_id": meta.get("collector_policy_id"),
        "termination_reason": meta.get("termination_reason"),
        "preprocessing": "LeRobot pi05 resize_with_pad_torch, RGB [0,1], float32, 224x224",
    }
    np.savez_compressed(
        destination,
        **arrays,
        state=state,
        frame_indices=frames[indices],
        timestamps=times,
        executed_steps=np.r_[0, np.diff(frames[indices])],
        task=np.array(meta["tasks"][0]),
        episode_id=np.array(f"{root.name}/episode_{episode:06d}"),
        provenance=np.array(json.dumps(provenance)),
        **labels,
    )
    print(f"SAVED {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episodes", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--base-camera", default="observation.images.right_environment_1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    for episode in args.episodes:
        prepare(
            args.dataset,
            episode,
            args.output,
            stride=args.stride,
            sample_count=args.sample_count,
            base_camera=args.base_camera,
        )


if __name__ == "__main__":
    main()
