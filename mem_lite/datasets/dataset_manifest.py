"""M1/M2共用的视频级划分、可迁移样本路径与训练清单读取。"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from mem_lite.datasets.m1_samples import read_annotation

MANIFEST_FORMAT = "mem-lite-dataset-v1"


def save_manifest(path, manifest):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_manifest(dataset_root, m1_config, m2_config, validation_fraction=0.2, split_seed=42,
                    main_view="cam_high"):
    """先按任务内的完整视频划分，再构造扰动；与目录位置和遍历顺序无关。"""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction需位于[0,1)")
    dataset_root = Path(dataset_root).resolve()
    videos, sources, identifiers = [], set(), set()
    for path in sorted(dataset_root.glob("*/annotations/*.annotations.json")):
        annotation = read_annotation(path)
        video = (path.parent.parent / annotation["video_file"]).resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        if len(annotation["stages"]) < 2:
            raise ValueError(f"M2需要至少两个标注区间：{path}")
        task = path.parent.parent.name
        video_id = f"{task}/{video.stem}"
        if video in sources or video_id in identifiers:
            raise ValueError(f"同一视频出现多份标注或输出标识重复：{video_id}")
        sources.add(video)
        identifiers.add(video_id)
        seeds = {stage: int.from_bytes(hashlib.sha256(
            f"{config['seed']}:{stage}:{video_id}".encode()).digest()[:4], "big")
            for stage, config in (("m1", m1_config), ("m2", m2_config))}
        videos.append({
            "video_id": video_id, "task": task, "video": video.relative_to(dataset_root).as_posix(),
            "annotation": path.relative_to(dataset_root).as_posix(),
            "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frame_count": annotation["total_frames"], "interval_count": len(annotation["stages"]),
            "seeds": seeds, "split": "train", "status": "pending",
            "samples": {stage: f"{stage}/{video_id}/samples.jsonl" for stage in ("m1", "m2")},
        })
    if not videos:
        raise ValueError("未找到已标注主视角视频")
    by_task = defaultdict(list)
    for video in videos:
        by_task[video["task"]].append(video)
    for task, group in sorted(by_task.items()):
        # 每任务独立随机流；新增其他任务不会改变本任务的划分。
        random.Random(f"{split_seed}:{task}").shuffle(group)
        count = (min(len(group) - 1, max(1, int(len(group) * validation_fraction + 0.5)))
                 if validation_fraction > 0 and len(group) > 1 else 0)
        for video in group[:count]:
            video["split"] = "validation"
    manifest = {
        "format_version": MANIFEST_FORMAT, "status": "planned", "dataset_root": str(dataset_root),
        "split": {"seed": split_seed, "validation_fraction": validation_fraction,
                  "strategy": "per-task whole-video; half-up rounding; singleton stays in train"},
        "main_view": main_view, "configs": {"m1": m1_config, "m2": m2_config}, "videos": videos,
    }
    manifest["summary"] = summarize_manifest(manifest)
    return manifest


def summarize_manifest(manifest):
    summary = {}
    for split in ("train", "validation"):
        group = [video for video in manifest["videos"] if video["split"] == split]
        by_task = defaultdict(int)
        for video in group:
            by_task[video["task"]] += 1
        counts = {key: sum(video.get("counts", {}).get(key, 0) for video in group)
                  for key in ("m1_samples", "m1_valid", "m1_invalid", "m1_skipped_empty", "m2_queues", "m2_prefixes")}
        summary[split] = {"videos": len(group), "completed_videos": sum(v["status"] == "complete" for v in group),
                          "source_frames": sum(v["frame_count"] for v in group), "by_task": dict(by_task), **counts}
    return summary


def load_sample_files(manifest_path, stage, split="train"):
    """样本路径相对清单解析；训练入口固定读取train，评测可显式读取validation。"""
    if stage not in ("m1", "m2") or split not in ("train", "validation"):
        raise ValueError("需要m1/m2阶段及train/validation划分")
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format_version"] != MANIFEST_FORMAT or manifest["status"] != "complete":
        raise ValueError("数据清单尚未完整构建，不能把计划或部分结果作为训练/评测集")
    paths = [(manifest_path.parent / video["samples"][stage]).resolve()
             for video in manifest["videos"] if video["split"] == split]
    if not paths:
        raise ValueError(f"{split}划分为空")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return [str(path) for path in paths]
