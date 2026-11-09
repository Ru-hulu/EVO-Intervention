"""Adjacent assembly rounds become one task: A/2 followed by (1+B)/2."""

import argparse
import concurrent.futures
import csv
import json
import math
import subprocess
from pathlib import Path

from mem_lite.datasets.prepare_recovery import CAMERA_DIRS, probe_video, sha256, write_json
from qwen_training.delta_to_progress import episode_to_progress

FORMAT = "streamforest-recovery-pairs-v1"
PROMPT = "连续完成两次插螺栓任务：每次拿起套筒和螺钉，将螺钉插入套筒，再将装配好的组件放入盒子。"
VALIDATION = ("file-000", "file-003", "file-008", "file-017")
REVIEW = ("file-019",)


def paired_progress(first, second, gap=0):
    """Preserve within-round rollback; intervening idle frames remain at 0.5."""
    if gap < 0 or not first or not second:
        raise ValueError("Rounds must be nonempty and must not overlap")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in (*first, *second)):
        raise ValueError("Single-round progress must be finite and within [0,1]")
    if first[-1] != 1.0 or second[-1] != 1.0:
        raise ValueError("Both annotated rounds must end at progress 1")
    return [p / 2 for p in first] + [0.5] * gap + [(1 + p) / 2 for p in second]


def prepare_index(dataset_root, output_dir, validation=VALIDATION, review=REVIEW):
    root, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    annotations = sorted((root / "splitlable").glob("*.JSON"))
    names = {p.stem for p in annotations}
    if set(validation) & set(review) or not (set(validation) | set(review)).issubset(names):
        raise ValueError("Invalid or overlapping source-video split")
    records, payloads, sources = [], [], []
    for path in annotations:
        video_id = path.stem
        original_videos = [root / camera / f"{video_id}.mp4" for camera in CAMERA_DIRS]
        metadata = [probe_video(video) for video in original_videos]
        if any(item != metadata[0] for item in metadata):
            raise ValueError(f"Unsynchronized camera views: {video_id}")
        delta_path = root / "delta" / f"{video_id}-delta.txt"
        deltas = [float(line.split()[0]) for line in delta_path.read_text().splitlines() if line.strip()]
        if len(deltas) != metadata[0]["frames"] or any(not math.isfinite(d) or d < 0 for d in deltas):
            raise ValueError(f"Invalid framewise deltas: {video_id}")
        annotation = json.loads(path.read_text())
        episodes = [e for task in annotation["tasks"] for e in task["episodes"]]
        if len(episodes) != 5 or [e["episode"] for e in episodes] != [1, 2, 3, 4, 5]:
            raise ValueError(f"Expected five ordered rounds: {video_id}")
        task_ids = [t["id"] for t in annotation["taskid"]]
        rounds = []
        for episode in episodes:
            stages = episode["subtasks"]
            start, end = stages[0]["30FPSstart_frame"], stages[-1]["30FPSend_frame"] + 1
            values, _ = episode_to_progress(deltas, stages, task_ids)
            rounds.append((start, end, values))
        split = "review" if video_id in review else "validation" if video_id in validation else "train"
        sources.append({"video_id": video_id, "split": split, **metadata[0],
                        "videos": [str(p) for p in original_videos],
                        "video_sha256": [sha256(p) for p in original_videos],
                        "annotation_sha256": sha256(path), "delta_sha256": sha256(delta_path)})
        for ordinal, (left, right) in enumerate(zip(rounds, rounds[1:]), 1):
            start, left_end, first = left
            boundary, end, second = right
            values = paired_progress(first, second, boundary - left_end)
            letters = "ABCDE"[ordinal-1:ordinal+1]
            episode_id = f"{video_id}/pair-{letters}"
            row = {"episode_id": episode_id, "source_video_id": video_id, "split": split,
                   "source_episodes": [ordinal, ordinal+1], "task_prompt": PROMPT,
                   "source_videos": [str(p) for p in original_videos],
                   "source_frame_range": [start, end], "frame_range": [0, end-start],
                   "first_round_end": left_end-start, "second_round_start": boundary-start,
                   "fps": metadata[0]["fps"], "progress_csv": f"labels/{episode_id}/progress.csv",
                   "videos": [str(output / "videos" / episode_id / f"{camera}.mp4") for camera in CAMERA_DIRS],
                   "label_status": "pending_review" if split == "review" else "ready",
                   "review_reason": "file-019 has a cropped late completion; excluded by user decision" if split == "review" else None,
                   "rollback_transitions": sum(b < a - 1e-8 for a, b in zip(values, values[1:]))}
            records.append(row)
            rows = []
            for local, frame in enumerate(range(start, end)):
                part = 1 if frame < left_end else 2 if frame >= boundary else 0
                single = first[frame-start] if part == 1 else second[frame-boundary] if part == 2 else 1.0
                rows.append((local, frame, part, single, deltas[frame], values[local]))
            payloads.append(rows)
    output.mkdir(parents=True, exist_ok=False)
    for row, values in zip(records, payloads, strict=True):
        target = output / row["progress_csv"]
        target.parent.mkdir(parents=True)
        with target.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "source_frame_index", "round_index", "round_pt", "delta", "pt"])
            writer.writerows(values)
        row["progress_sha256"] = sha256(target)
    for split in ("train", "validation", "review", "all"):
        selected = records if split == "all" else [row for row in records if row["split"] == split]
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    manifest = {"format_version": FORMAT, "dataset_root": str(root), "sources": sources,
                "progress_method": "first=single_pt/2; second=(1+single_pt)/2; idle=0.5",
                "single_progress_method": "qwen_training.delta_to_progress.episode_to_progress",
                "splits": {split: [s["video_id"] for s in sources if s["split"] == split]
                           for split in ("train", "validation", "review")},
                "episode_counts": {split: sum(row["split"] == split for row in records)
                                   for split in ("train", "validation", "review")},
                "num_pairs": len(records), "frame_range_convention": "[start,end)",
                "memory_reset": "once at pair start; continuous across both original rounds",
                "review_excluded_from_training_and_validation": True}
    write_json(output / "manifest.json", manifest)
    return manifest


def read_records(prepared):
    prepared = Path(prepared)
    if json.loads((prepared / "manifest.json").read_text())["format_version"] != FORMAT:
        raise ValueError("Not a paired-recovery index")
    rows = [json.loads(line) for line in (prepared / "all.jsonl").read_text().splitlines()]
    for row in rows:
        if sha256(prepared / row["progress_csv"]) != row["progress_sha256"]:
            raise ValueError("Progress labels changed after indexing")
    return rows


def clip_videos(prepared_dir, workers=6):
    prepared = Path(prepared_dir)
    jobs = [(row, source, target) for row in read_records(prepared)
            for source, target in zip(row["source_videos"], row["videos"], strict=True)]
    def clip(job):
        row, source, target = job
        target = Path(target)
        expected = {"frames": row["frame_range"][1], "fps": row["fps"], "start_time": 0.0}
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".partial.mp4")
            start, end = row["source_frame_range"]
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "2", "-i", source,
                            "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS", "-an",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", "2",
                            "-vsync", "0", "-movflags", "+faststart", str(temporary)], check=True)
            if probe_video(temporary) != expected:
                raise ValueError(f"Clip frame/time mismatch: {target}")
            temporary.replace(target)
        if probe_video(target) != expected:
            raise ValueError(f"Existing clip mismatch: {target}")
        return {"path": str(target), "sha256": sha256(target), **expected}
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for item in pool.map(clip, jobs):
            results.append(item)
            if len(results) % 12 == 0:
                print(json.dumps({"clips_completed": len(results), "total_clips": len(jobs)}), flush=True)
    write_json(prepared / "clips-report.json", {"num_clips": len(results), "clips": results})


def cache_worker(prepared_dir, model_path, cache_dir, worker=0, workers=1, device="cuda"):
    import fcntl
    import torch
    from mem_lite.datasets.prepare_stream import DEFAULT_MEMORY_CONFIG, build_episode, iter_observations
    from mem_lite.memory.stream import StreamMemory
    from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder

    torch.set_num_threads(2)
    prepared, output = Path(prepared_dir), Path(cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / f"worker-{worker}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = read_records(prepared)[worker::workers]
    memory = json.loads(DEFAULT_MEMORY_CONFIG.read_text())
    visual, processor, vision_hash = load_vision_encoder(model_path, device)
    stream = StreamMemory(QwenFrameEncoder(visual), processor, memory, 65536, 4096)
    origin = {"method": "streamforest-v1", "vision_sha256": vision_hash, "memory_config": memory,
              "sample_fps": 5.0, "include_episode_last_frame": True, "max_pixels": 65536, "min_pixels": 4096,
              "view_policy": "main-first; missing auxiliary views use RGB black 64x64; v1"}
    for i, row in enumerate(rows, 1):
        target = output / row["split"] / row["episode_id"]
        if (target / "summary.json").is_file():
            if json.loads((target / "summary.json").read_text())["source"] != row:
                raise ValueError("Existing cache belongs to another index")
            continue
        with (prepared / row["progress_csv"]).open() as handle:
            progress = {int(r["frame_index"]): float(r["pt"]) for r in csv.DictReader(handle)}
        observations = iter_observations(row["videos"], progress, sample_fps=5.0,
                                        start_frame=0, end_frame=row["frame_range"][1])
        result = build_episode(observations, stream, target, row["task_prompt"], row["episode_id"], origin, source=row)
        print(json.dumps({"worker": worker, "completed": i, "total": len(rows), "episode": row["episode_id"],
                          "samples": result["num_samples"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index")
    index.add_argument("--dataset-root", required=True)
    index.add_argument("--output-dir", required=True)
    clips = commands.add_parser("clips")
    clips.add_argument("--prepared-dir", required=True)
    clips.add_argument("--workers", type=int, default=6)
    cache = commands.add_parser("cache")
    cache.add_argument("--prepared-dir", required=True)
    cache.add_argument("--model-path", required=True)
    cache.add_argument("--cache-dir", required=True)
    cache.add_argument("--worker", type=int, default=0)
    cache.add_argument("--workers", type=int, default=1)
    cache.add_argument("--device", default="cuda")
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"index": prepare_index, "clips": clip_videos, "cache": cache_worker}[command](**args)
    if result:
        print(json.dumps(result["episode_counts"]))


if __name__ == "__main__":
    main()
