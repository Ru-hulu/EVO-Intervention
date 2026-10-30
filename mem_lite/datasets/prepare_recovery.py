"""恢复任务：三视角校验、单轮回退进度、按原视频划分，以及独立记忆缓存。"""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

from qwen_training.delta_to_progress import episode_to_progress

EXCLUDED_VIDEOS = ("file-019",)
CAMERA_DIRS = ("30FPS", "left_wrist", "right_wrist")
TASK_PROMPT = "拿起套筒和螺钉，将螺钉插入套筒，再将装配好的组件放入盒子。"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path):
    import av

    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        if not stream.frames or not stream.average_rate:
            raise ValueError(f"视频缺少帧数或帧率：{path}")
        return {"frames": stream.frames, "fps": float(stream.average_rate),
                "start_time": float((stream.start_time or 0) * stream.time_base)}


def prepare_index(dataset_root, output_dir, validation_videos=4, seed=42):
    root, output = Path(dataset_root).resolve(), Path(output_dir).resolve()
    labels = sorted(path for path in (root / "splitlable").glob("*.JSON")
                    if path.stem not in EXCLUDED_VIDEOS)
    if not 0 < validation_videos < len(labels):
        raise ValueError("验证视频数必须大于0，并且保留至少一个训练视频")
    validation = set(random.Random(seed).sample([path.stem for path in labels], validation_videos))
    prepared, sources = [], []
    # 全部输入校验通过后再创建输出，避免把错误输入包装成可用训练集。
    for path in labels:
        video_id = path.stem
        videos = [root / camera / f"{video_id}.mp4" for camera in CAMERA_DIRS]
        for video in videos:
            if not video.is_file():
                raise FileNotFoundError(f"缺少真实视角视频：{video}")
        metadata = [probe_video(video) for video in videos]
        frames, fps = metadata[0]["frames"], metadata[0]["fps"]
        if any(item["frames"] != frames or abs(item["fps"] - fps) > 1e-3
               or abs(item["start_time"] - metadata[0]["start_time"]) > 1e-6 for item in metadata):
            raise ValueError(f"{video_id} 三视角帧数、帧率或起始时间不同")
        annotation = json.loads(path.read_text(encoding="utf-8"))
        delta_path = root / "delta" / f"{video_id}-delta.txt"
        deltas = [float(line.split()[0]) for line in delta_path.read_text().splitlines() if line.strip()]
        if len(deltas) != frames or any(not math.isfinite(value) or value < 0 for value in deltas):
            raise ValueError(f"{video_id} delta 必须与视频逐帧对应且为有限非负数")
        task_ids = [item["id"] for item in annotation["taskid"]]
        split = "validation" if video_id in validation else "train"
        source = {"video_id": video_id, "split": split, "frames": frames, "fps": fps,
                  "videos": [str(video) for video in videos],
                  "video_sha256": [sha256(video) for video in videos],
                  "annotation_sha256": sha256(path), "delta_sha256": sha256(delta_path)}
        sources.append(source)
        previous_end = 0
        episodes = [episode for task in annotation["tasks"] for episode in task["episodes"]]
        if not episodes:
            raise ValueError(f"{video_id} 没有可训练的单轮标注")
        for ordinal, episode in enumerate(episodes, 1):
            subtasks = episode["subtasks"]
            start, end = subtasks[0]["30FPSstart_frame"], subtasks[-1]["30FPSend_frame"] + 1
            if start < previous_end:
                raise ValueError(f"{video_id} 单轮时间范围重叠或顺序错误")
            values, _ = episode_to_progress(deltas, subtasks, task_ids)
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError(f"{video_id} 回退进度必须为 [0,1] 的有限值")
            if not math.isclose(values[-1], 1.0):
                raise ValueError(f"{video_id} 单轮结束进度不为1")
            episode_id = f"{video_id}/episode-{ordinal:02d}"
            row = {"episode_id": episode_id, "source_video_id": video_id,
                   "annotation_episode": episode.get("episode", ordinal), "split": split,
                   "videos": source["videos"], "source_frame_range": [start, end],
                   "fps": fps, "task_prompt": TASK_PROMPT,
                   "progress_csv": f"episodes/{episode_id}/progress.csv",
                   "rollback_transitions": sum(b < a - 1e-8 for a, b in zip(values, values[1:])),
                   "annotation_sha256": source["annotation_sha256"],
                   "delta_sha256": source["delta_sha256"]}
            prepared.append((row, deltas[start:end], values))
            previous_end = end
    output.mkdir(parents=True, exist_ok=False)
    records = {"train": [], "validation": []}
    for row, deltas, values in prepared:
        csv_path = output / row["progress_csv"]
        csv_path.parent.mkdir(parents=True)
        start, end = row["source_frame_range"]
        with csv_path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target)
            writer.writerow(["frame_index", "delta", "pt"])
            writer.writerows(zip(range(start, end), deltas, values, strict=True))
        row["progress_sha256"] = sha256(csv_path)
        records[row["split"]].append(row)
    for split, rows in records.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"format_version": "streamforest-recovery-index-v1", "dataset_root": str(root),
                "excluded_videos": list(EXCLUDED_VIDEOS), "seed": seed,
                "progress_method": "qwen_training.delta_to_progress.episode_to_progress",
                "frame_range_convention": "[start_frame, end_frame)",
                "memory_reset": "at every episode; no observations outside its frame range",
                "splits": {split: sorted({row["source_video_id"] for row in rows})
                           for split, rows in records.items()},
                "episode_counts": {split: len(rows) for split, rows in records.items()},
                "rollback_transitions": sum(row["rollback_transitions"] for row, _, _ in prepared),
                "sources": sources}
    write_json(output / "manifest.json", manifest)
    return manifest


def build_caches(prepared_dir, model_path, cache_dir, device="cuda", sample_fps=5.0,
                 max_pixels=65536, min_pixels=4096, memory_config=None):
    from mem_lite.datasets.prepare_stream import DEFAULT_MEMORY_CONFIG, build_episode, iter_observations
    from mem_lite.memory.stream import StreamMemory
    from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder

    prepared, output = Path(prepared_dir).resolve(), Path(cache_dir).resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["format_version"] != "streamforest-recovery-index-v1":
        raise ValueError("不是恢复任务的单轮索引")
    splits = manifest["splits"]
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("训练和验证不能共享源视频")
    records = {}
    for split in ("train", "validation"):
        rows = [json.loads(line) for line in (prepared / f"{split}.jsonl").read_text().splitlines()]
        for row in rows:
            if row["source_video_id"] in EXCLUDED_VIDEOS or row["source_video_id"] not in splits[split]:
                raise ValueError("索引包含排除视频或属于错误划分")
            if sha256(prepared / row["progress_csv"]) != row["progress_sha256"]:
                raise ValueError("进度 CSV 已变化，请重新生成单轮索引")
        records[split] = rows
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(memory_config or DEFAULT_MEMORY_CONFIG).read_text())
    visual, processor, vision_hash = load_vision_encoder(model_path, device)
    stream = StreamMemory(QwenFrameEncoder(visual), processor, config, max_pixels, min_pixels)
    origin = {"method": "streamforest-v1", "vision_sha256": vision_hash, "memory_config": config,
              "sample_fps": sample_fps, "include_episode_last_frame": True,
              "max_pixels": max_pixels, "min_pixels": min_pixels}
    sample_files = {}
    for split, rows in records.items():
        sample_files[split] = []
        for row in rows:
            with (prepared / row["progress_csv"]).open() as source:
                progress = {int(item["frame_index"]): float(item["pt"]) for item in csv.DictReader(source)}
            start, end = row["source_frame_range"]
            directory = output / split / row["episode_id"]
            observations = iter_observations(row["videos"], progress, sample_fps=sample_fps,
                                            start_frame=start, end_frame=end)
            result = build_episode(observations, stream, directory, row["task_prompt"],
                                   row["episode_id"], origin, source=row)
            sample_files[split].append(str(directory))
            print(json.dumps({"episode": row["episode_id"], "split": split,
                              "samples": result["num_samples"]}), flush=True)
        write_json(output / f"{split}.sample_files.json", sample_files[split])
    train_config = json.loads((Path(__file__).parents[1] / "configs/stage3_train.json").read_text())
    train_config.update(model_path=str(Path(model_path).resolve()), sample_files=sample_files["train"],
                        max_pixels=max_pixels, output_dir=str(output / "training"))
    write_json(output / "train.config.json", train_config)
    write_json(output / "manifest.json", {"source_index": str(prepared), **manifest,
                                          "history_source": origin, "sample_files": sample_files})
    return sample_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="CPU：校验三视角、生成回退进度和按原视频划分")
    index.add_argument("--dataset-root", required=True)
    index.add_argument("--output-dir", required=True)
    index.add_argument("--validation-videos", type=int, default=4)
    index.add_argument("--seed", type=int, default=42)
    cache = commands.add_parser("cache", help="冻结视觉推理：每轮单独构造 StreamForest 历史")
    cache.add_argument("--prepared-dir", required=True)
    cache.add_argument("--model-path", required=True)
    cache.add_argument("--cache-dir", required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--sample-fps", type=float, default=5.0)
    cache.add_argument("--max-pixels", type=int, default=65536)
    cache.add_argument("--min-pixels", type=int, default=4096)
    cache.add_argument("--memory-config")
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = prepare_index(**args) if command == "index" else build_caches(**args)
    if command == "index":
        print(json.dumps({key: result[key] for key in ("splits", "episode_counts", "excluded_videos",
                                                      "rollback_transitions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
