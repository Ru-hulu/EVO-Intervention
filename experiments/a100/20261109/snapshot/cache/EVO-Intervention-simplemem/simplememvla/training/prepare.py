"""Audit a pinned ProPriMem metadata snapshot and preserve its frozen splits."""

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from qwen_training.datasets.proprimem.policies import POLICIES
from qwen_training.datasets.proprimem.prepare import annotation_events


def progress_labels(annotation, task_id):
    total = annotation["total_frames"]
    if annotation.get("frame_index_base") != 0 or annotation.get("frame_range_convention") != "[start_frame, end_frame)":
        raise ValueError("unsupported_frame_convention")
    cursor = 0
    for stage in annotation["stages"]:
        if stage["start_frame"] != cursor or not cursor < stage["end_frame"] <= total:
            raise ValueError("invalid_stage_coverage")
        cursor = stage["end_frame"]
    if cursor != total or len(annotation["stages"]) != annotation["num_stages"]:
        raise ValueError("incomplete_stage_coverage")
    if "progress" in annotation:
        values = annotation["progress"]["dense_progress"]
        kind = annotation["progress"]["policy"]
        if len(values) != total or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("invalid_dense_progress")
        return dict(kind=kind, dense=values)
    if task_id not in POLICIES:
        raise ValueError("unsupported_annotation_semantics")
    events = annotation_events(annotation, POLICIES[task_id])
    # LIBERO boundaries are exclusive in this snapshot, like all other intervals.
    ends = [frame - (POLICIES[task_id][0] == "libero") for _, frame in events]
    if (
        not ends
        or any(not 0 <= x < total for x in ends)
        or any(b <= a for a, b in pairwise(ends))
    ):
        raise ValueError("invalid_completion_boundaries")
    return dict(
        kind="derived_event_normalized",
        ends=ends,
        unit="source_action_segment" if POLICIES[task_id][0] == "source_segments" else "annotated_action",
    )


def audit(root, out, revision, media_root=None):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    tasks = {x["task_path"]: x for x in manifest["tasks"]}
    records, report = [], dict(revision=revision, tasks=[], excluded=[], extras=[], warnings=[])
    hashes = defaultdict(list)
    for path in sorted(root.glob("*/*/metadata.json")):
        meta = json.loads(path.read_text())
        rel = path.parent.relative_to(root).as_posix()
        if rel not in tasks:
            report["warnings"].append(dict(task=rel, reason="not_in_frozen_manifest"))
            continue
        split = {}
        for name in ("train", "val"):
            for sample in tasks[rel][name]:
                if sample in split:
                    raise ValueError(f"overlapping frozen splits: {rel}/{sample}")
                split[sample] = name
        rows = [json.loads(line) for line in (path.parent / meta["trajectory_index"]).read_text().splitlines() if line]
        counts = Counter(total=len(rows))
        seen = set()
        for row in rows:
            sample = row["sample_id"]
            key = f"{rel}/{sample}"
            reasons, warnings = [], []
            if sample in seen:
                raise ValueError(f"duplicate sample: {key}")
            seen.add(sample)
            if sample not in split:
                reasons.append("missing_frozen_split")
            instruction = row.get("source_instruction")
            if not instruction:
                reasons.append("missing_source_instruction")
            progress = None
            if row.get("has_annotation"):
                counts["annotated"] += 1
                try:
                    ann = json.loads((path.parent / row["annotation_path"]).read_text())
                    if ann["total_frames"] != row["total_frames"] or ann["fps"] != row["fps"]:
                        raise ValueError("annotation_video_timebase_mismatch")
                    progress = progress_labels(ann, meta["task_id"])
                    counts["progress_candidate"] += 1
                except (KeyError, ValueError, OSError) as error:
                    reasons.append(f"annotation: {error}")
            else:
                counts["unannotated"] += 1
            success = row.get("task_success")
            if success is not None and type(success) is not bool:
                reasons.append("unsupported_success_label")
                success = None
            if success is not None:
                counts["explicit_success"] += 1
            if progress is None and success is None:
                reasons.append("no_progress_or_success_supervision")
            if row.get("has_retry"):
                warnings.append("retry_present_monotonic_progress_may_be_wrong")
            if progress and progress.get("unit") == "source_action_segment":
                warnings.append("source_segments_are_not_physical_repetitions")
            views = row.get("video_views", {})
            order = [name for name in ("main_view", "left_view", "right_view") if name in views]
            order += sorted(set(views) - set(order))
            if not order:
                reasons.append("no_video_views")
            paths = [str(Path(rel) / views[name]["video_path"]) for name in order]
            sizes = [views[name].get("video_size_bytes") for name in order]
            if any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
                raise ValueError(f"unsafe video path: {key}")
            if media_root:
                for file, size in zip(paths, sizes, strict=True):
                    local = Path(media_root) / file
                    if not local.is_file():
                        reasons.append("media_missing")
                    elif size is not None and local.stat().st_size != size:
                        reasons.append("media_size_mismatch")
            counts[f"views_{len(order)}"] += 1
            record = dict(
                id=key,
                task=rel,
                task_id=meta["task_id"],
                split=split.get(sample),
                instruction=instruction,
                fps=row["fps"],
                total_frames=row["total_frames"],
                videos=paths,
                video_sizes=sizes,
                progress=progress,
                success=success,
                warnings=warnings,
            )
            if reasons:
                report["excluded"].append(dict(id=key, reasons=sorted(set(reasons))))
            else:
                records.append(record)
                counts["eligible"] += 1
                sha = row.get("video_sha256")
                if sha:
                    hashes[sha].append(record)
        if set(split) != seen:
            report["warnings"].append(
                dict(
                    task=rel,
                    reason="manifest_index_membership_mismatch",
                    missing=sorted(set(split) - seen),
                    extra=sorted(seen - set(split)),
                )
            )
        report["tasks"].append(dict(task=rel, **counts))
    leaked = {x["id"] for group in hashes.values() if len({x["split"] for x in group}) > 1 for x in group}
    for key in sorted(leaked):
        report["excluded"].append(dict(id=key, reasons=["exact_primary_video_cross_split"]))
    records = [x for x in records if x["id"] not in leaked]
    for p in sorted(root.glob("*/multiview_manifest.json")):
        m = json.loads(p.read_text())
        annotations = list((p.parent / "splitlable").glob("*.JSON"))
        report["extras"].append(
            dict(
                dataset=p.parent.name,
                episodes=len(m["episodes"]),
                annotations=len(annotations),
                reason="separate_cycle_recovery_schema_and_no_core_frozen_split_adapter",
            )
        )
    report["summary"] = dict(
        indexed=sum(x["total"] for x in report["tasks"]),
        eligible=len(records),
        train=sum(x["split"] == "train" for x in records),
        val=sum(x["split"] == "val" for x in records),
        excluded=len(report["excluded"]),
        explicit_success=sum(x["success"] is not None for x in records),
        preference_pairs=0,
    )
    (out / "records.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--media-root")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.out, args.revision, args.media_root)["summary"]))


if __name__ == "__main__":
    main()
