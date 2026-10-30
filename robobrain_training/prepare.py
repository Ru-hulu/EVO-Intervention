"""复用 5 FPS 原始图像和 PT，构建独立的 RoboBrain SFT 数据目录。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import av
from PIL import Image

from .data import VIEWS, make_sample


def sample_indices(count, fps, sample_fps=5):
    step = Fraction(str(fps)) / Fraction(str(sample_fps))
    indices, index = [], 0
    while math.ceil(index * step) < count:
        indices.append(math.ceil(index * step))
        index += 1
    return indices


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def prepare_episode(row, raw_root, frames_root, labels_root, output):
    episode, split = row["video_id"], row["split"]
    annotation = json.loads((raw_root / row["annotation"]).read_text())
    cache = frames_root / split / episode
    source_samples = [json.loads(line) for line in (cache / "samples.jsonl").read_text().splitlines()]
    label_path = labels_root / episode / f"{Path(episode).name}.progress.csv"
    with label_path.open() as stream:
        progress = {int(x["frame_index"]): float(x["pt"]) for x in csv.DictReader(stream)}
    count = annotation["total_frames"]
    if set(progress) != set(range(count)) or progress[0] != 0 or progress[count - 1] != 1:
        raise ValueError(f"Incomplete PT labels or unexpected endpoints: {episode}")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in progress.values()):
        raise ValueError(f"Invalid PT labels: {episode}")
    destination = output / "images" / episode
    destination.mkdir(parents=True)
    linked = copied = 0
    image_sets = {}
    for sample in source_samples:
        frame_index = sample["frame_index"]
        if abs(sample["pt"] - progress[frame_index]) > 1e-8:
            raise ValueError(f"Cached PT differs from source CSV: {episode}/{frame_index}")
        paths = []
        for view in VIEWS:
            source = cache / sample["images"][view]
            target = destination / view / f"{frame_index:08d}.png"
            target.parent.mkdir(exist_ok=True)
            try:
                os.link(source, target)
                linked += 1
            except OSError:
                shutil.copy2(source, target)
                copied += 1
            paths.append(str(target.resolve()))
        image_sets[frame_index] = paths

    # 最后一张采样图未必是视频末帧；目标参考必须单独读取真实末帧。
    video_path = raw_root / row["video"]
    with av.open(str(video_path)) as source:
        video_stream = source.streams.video[0]
        video_stream.codec_context.thread_count = 2
        fps = video_stream.average_rate
        actual_count = 0
        middle = source_samples[len(source_samples) // 2]["frame_index"]
        for frame_index, frame in enumerate(source.decode(video=0)):
            if frame_index in (0, middle):
                with Image.open(image_sets[frame_index][0]) as cached:
                    if cached.convert("RGB").tobytes() != frame.to_image().tobytes():
                        raise ValueError(f"Sample image does not match original video: {episode}/{frame_index}")
            last_frame = frame
            actual_count += 1
    if actual_count != count or list(image_sets) != sample_indices(count, fps):
        raise ValueError(f"Video count or 5 FPS sample grid mismatch: {episode}")
    end_path = destination / "end_cam_high.png"
    last_frame.to_image().save(end_path)
    start = image_sets[0]
    samples = [make_sample(episode, index, annotation["task_prompt"], progress[index],
                           start, current, str(end_path.resolve())) for index, current in image_sets.items()]
    metadata = {"episode_id": episode, "split": split, "fps": float(fps), "raw_frame_count": count,
                "start_frame_index": 0, "end_frame_index": count - 1, "sample_count": len(samples),
                "labels_source": str(label_path), "labels_sha256": sha256(label_path),
                "annotation_source": str(raw_root / row["annotation"]),
                "linked_images": linked, "copied_images": copied}
    return samples, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--frames-root", required=True, type=Path)
    parser.add_argument("--labels-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = json.loads(args.source_manifest.read_text())
    selected = [row for row in source["videos"] if row["split"] in ("train", "validation")]
    rows_by_split = {"train": [], "validation": []}
    episodes = []
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare_episode, row, Path(source["dataset_root"]), args.frames_root,
                                   args.labels_root, output) for row in selected]
            for future in as_completed(futures):
                samples, metadata = future.result()
                rows_by_split[metadata["split"]].extend(samples)
                episodes.append(metadata)
                status = {"phase": "building", "completed_videos": len(episodes), "total_videos": len(selected)}
                write_json(output / "status.json", status)
                print(json.dumps(status), flush=True)
        for split, rows in rows_by_split.items():
            rows.sort(key=lambda row: (row["episode_id"], row["frame_index"]))
            with (output / f"{split}.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "format": "robobrain-start-current-end-v1", "status": "complete", "sample_fps": 5,
            "source_manifest": str(args.source_manifest), "source_manifest_sha256": sha256(args.source_manifest),
            "model": "BAAI/RoboBrain2.5-4B", "conditioning": "same-video start/current/end; supplied end-state reference",
            "image_order": ["reference_start_high", "reference_end_high", "before_start_high", "before_start_left",
                            "before_start_right", "after_current_high", "after_current_left", "after_current_right"],
            "label": "existing PT times 100; signed percentage with one decimal place",
            "expert_videos_excluded": sum(row["split"] == "expert" for row in source["videos"]),
            "splits": {split: {"samples": len(rows), "videos": len({row["episode_id"] for row in rows}),
                               "sha256": sha256(output / f"{split}.jsonl")} for split, rows in rows_by_split.items()},
            "episodes": sorted(episodes, key=lambda row: row["episode_id"]),
            "elapsed_seconds": time.monotonic() - started,
            "training_started": False,
        }
        write_json(output / "manifest.json", manifest)
        write_json(output / "status.json", {"phase": "completed", "splits": manifest["splits"], "training_started": False})
    except Exception as error:
        write_json(output / "status.json", {"phase": "failed", "error": str(error), "training_started": False})
        raise


if __name__ == "__main__":
    main()
