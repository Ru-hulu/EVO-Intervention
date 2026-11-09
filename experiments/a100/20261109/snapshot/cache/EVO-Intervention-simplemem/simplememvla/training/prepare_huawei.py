"""Import Huawei's existing progress CSVs and frozen splits without relabeling."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from qwen_training.datasets.proprimem.current_actions import normalize_current_actions, normalize_recovery_actions


def import_labels(core, recovery, media_root, out):
    core, recovery, media_root, out = map(Path, (core, recovery, media_root, out))
    selected = {}
    for split in ("train", "validation"):
        for directory in json.loads((core / f"{split}.sample_files.json").read_text()):
            source_id = Path(directory).relative_to(core / "cache" / split).as_posix()
            if source_id in selected:
                raise ValueError(f"Duplicate or leaking source: {source_id}")
            selected[source_id] = split
    core_sources = json.loads((core / "index.json").read_text())["sources"]
    metadata_paths = [core / "labels" / key / "metadata.json" for key in sorted(selected)]
    recovery_metadata = sorted((recovery / "labels/insert-screw-recover").glob("*/metadata.json"))
    if not recovery_metadata:
        raise FileNotFoundError(f"No recovery label metadata in {recovery}")
    metadata_paths += recovery_metadata
    records = []
    parents = {}
    for path in metadata_paths:
        meta = json.loads(path.read_text())
        source = meta["source"]
        task, parent = source["task"], source["source_id"]
        if parent == "insert-screw-recover/file-019":
            continue  # Preserve the existing user-approved exclusion.
        split = selected[parent] if task != "insert-screw-recover" else source["split"]
        if split != source["split"] or split not in ("train", "validation"):
            raise ValueError(f"Unexpected frozen split: {parent}")
        videos = []
        for original in source["videos"]:
            if original is not None:
                # Labels predate the dataset move; retain the task-relative camera path.
                suffix = original.split(f"/{task}/", 1)[1]
                video = media_root / task / suffix
                if not video.is_file():
                    raise FileNotFoundError(video)
                videos.append(str(video.resolve()))
        annotation = None
        annotation_digest = None
        actions = None
        annotation_path = media_root / task / "annotations" / f"{Path(parent).name}.json"
        if source.get("annotation"):
            annotation_path = media_root / task / source["annotation"].split(f"/{task}/", 1)[1]
        if source.get("annotation") or annotation_path.is_file():
            annotation_bytes = annotation_path.read_bytes()
            annotation_digest = hashlib.sha256(annotation_bytes).hexdigest()
            annotation = json.loads(annotation_bytes)
            if task != "insert-screw-recover":
                task_meta = json.loads((media_root / task / "metadata.json").read_text())
                actions = normalize_current_actions(annotation, task_meta["task_id"])
        for unit in meta["units"]:
            start, stop = unit["frame_range"]
            csv_path = path.parent / "progress.csv"
            # Ordinary units live beside metadata; recovery units have episode subdirectories.
            if task == "insert-screw-recover":
                csv_path = path.parent / unit["unit_id"].rsplit("/", 1)[1] / "progress.csv"
            content = csv_path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != unit["progress_sha256"]:
                raise ValueError(f"Progress checksum differs: {csv_path}")
            values = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
            if (
                not 0 <= start < stop <= source["frames"]
                or values.shape != (stop - start, 3)
                or not np.array_equal(values[:, 0], np.arange(start, stop))
                or not np.isfinite(values).all()
                or np.any((values[:, 2] < 0) | (values[:, 2] > 1))
            ):
                raise ValueError(f"Invalid source-frame progress: {csv_path}")
            unit_actions = actions
            if task == "insert-screw-recover" and annotation is not None:
                episodes = [episode for item in annotation["tasks"] for episode in item["episodes"]]
                episode_number = int(unit["unit_id"].rsplit("-", 1)[1])
                unit_actions = normalize_recovery_actions(episodes[episode_number - 1])
            if unit_actions is not None:
                intervals = unit_actions["intervals"]
                if intervals[0]["start_frame"] != start or intervals[-1]["end_frame"] != stop:
                    raise ValueError(f"Current-action and progress source ranges differ: {unit['unit_id']}")
                unit_actions = dict(unit_actions, annotation_sha256=annotation_digest)
            records.append(
                dict(
                    id=unit["unit_id"],
                    task=task,
                    source_id=parent,
                    split="val" if split == "validation" else "train",
                    instruction=source["task_prompt"],
                    fps=source["fps"],
                    total_frames=source["frames"],
                    frame_start=start,
                    frame_stop=stop,
                    videos=videos,
                    video_sizes=[Path(v).stat().st_size for v in videos],
                    progress=dict(kind=source["pt_method"], csv=str(csv_path.resolve()), sha256=digest),
                    success=None,
                    current_actions=unit_actions,
                )
            )
            parents.setdefault(parent, set()).add(split)
    if not records or any(len(splits) != 1 for splits in parents.values()):
        raise ValueError("Empty dataset or source-video split leakage")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Duplicate training unit")
    report = dict(
        records=len(records),
        tasks=len({r["task"] for r in records}),
        splits=dict(Counter(r["split"] for r in records)),
        task_splits={
            task: dict(Counter(r["split"] for r in records if r["task"] == task))
            for task in sorted({r["task"] for r in records})
        },
        excluded_core=[r["source_id"] for r in core_sources if r["source_id"] not in selected],
        excluded_recovery=["insert-screw-recover/file-019"],
        sources=dict(core=str(core), recovery=str(recovery)),
        current_action_coverage=dict(
            Counter(
                r["current_actions"]["provenance"] if r["current_actions"] else "missing_boundaries" for r in records
            )
        ),
        progress_recomputed=False,
        success_supervision=False,
        preference_supervision=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", default="/cache/cvpzhr/datasets/forest-proprimem-cap50-5fps-20261103")
    parser.add_argument("--recovery", default="/cache/cvpzhr/datasets/forest-proprimem-5fps-20261030")
    parser.add_argument("--media-root", default="/cache/EVO-Intervention/datasets/ProPriMem-current")
    parser.add_argument("--out", default="outputs/proprimem-huawei-stage1")
    print(json.dumps(import_labels(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
