#!/usr/bin/env python3
"""Export explicitly selected LeRobot v3 recordings to ProPriMem/Order, offline."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


CAMERAS = {
    "main_view": "observation.images.right_environment_1",
    "left_view": "observation.images.left_wrist",
    "right_view": "observation.images.right_wrist",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Path outside source root: {relative}")
    return path


def load_plan(manifest, source_root):
    import pyarrow.parquet as pq

    plan, ids, sources_seen = [], set(), set()
    for task in manifest["tasks"]:
        if not re.fullmatch(r"O\d{2,}", task["task_id"]) or not re.fullmatch(r"[a-z0-9_]+", task["task_slug"]):
            raise ValueError("Invalid Order task ID/slug")
        if task["task_id"] in ids:
            raise ValueError("Duplicate task ID")
        ids.add(task["task_id"])
        episodes, snapshots = [], {}
        for source in task["sources"]:
            name = source["dataset"]
            if name in sources_seen:
                raise ValueError(f"Duplicate source dataset: {name}")
            sources_seen.add(name)
            root = child(source_root, name)
            info_path = root / "meta/info.json"
            info = json.loads(info_path.read_text())
            if info["codebase_version"] != "v3.0" or info["total_episodes"] != source["episodes"]:
                raise ValueError(f"Source version/count changed: {name}")
            if info["total_episodes"] <= 0 or info["fps"] <= 0:
                raise ValueError(f"Empty/invalid source: {name}")
            if any(info["features"].get(camera, {}).get("dtype") != "video" for camera in CAMERAS.values()):
                raise ValueError(f"Missing configured three-view cameras: {name}")
            meta_files = [info_path] + sorted((root / "meta/episodes").rglob("*.parquet"))
            rows = [row for p in meta_files[1:] for row in pq.read_table(p).to_pylist()]
            rows.sort(key=lambda row: row["episode_index"])
            if [r["episode_index"] for r in rows] != list(range(source["episodes"])):
                raise ValueError(f"Noncontiguous or unfinished episode metadata: {name}")
            if sum(r["length"] for r in rows) != info["total_frames"]:
                raise ValueError(f"Source frame total mismatch: {name}")
            snapshots.update({str(p): sha256(p) for p in meta_files})
            for row in rows:
                instructions = row["tasks"]
                if len(instructions) != 1 or not instructions[0].strip() or row["length"] <= 0:
                    raise ValueError(f"Expected one nonempty instruction per episode: {name}")
                views = {}
                for view, camera in CAMERAS.items():
                    prefix = f"videos/{camera}"
                    start = row[prefix + "/from_timestamp"] * info["fps"]
                    end = row[prefix + "/to_timestamp"] * info["fps"]
                    if abs(start - round(start)) > 0.01 or abs(end - start - row["length"]) > 0.01:
                        raise ValueError(f"Non-frame-aligned range: {name}, {row['episode_index']}, {view}")
                    relative = info["video_path"].format(
                        video_key=camera, chunk_index=row[prefix + "/chunk_index"],
                        file_index=row[prefix + "/file_index"])
                    path = child(root, relative)
                    if not path.is_file():
                        raise ValueError(f"Missing video: {path}")
                    views[view] = dict(path=path, source_path=f"{name}/{relative}", start=round(start))
                episodes.append(dict(source_dataset=name, source_episode=row["episode_index"],
                                     instruction=instructions[0], frames=row["length"],
                                     fps=info["fps"], views=views))
        median = statistics.median(e["frames"] / e["fps"] for e in episodes)
        warnings = [dict(dataset=e["source_dataset"], episode=e["source_episode"],
                         duration_s=e["frames"] / e["fps"], reason="less than 60% of task median; review only")
                    for e in episodes if e["frames"] / e["fps"] < median * 0.6]
        plan.append(dict(task=task, episodes=episodes, snapshots=snapshots, warnings=warnings))
    return plan


def export_clip(source, target, start, count, fps):
    """Decode from the preceding keyframe, cut exact frames, encode losslessly."""
    import av

    rate = Fraction(fps).limit_denominator()
    digest, written = hashlib.sha256(), 0
    with av.open(str(source)) as inp, av.open(str(target), "w") as out:
        video = inp.streams.video[0]
        if video.average_rate != rate or video.codec_context.pix_fmt != "yuv420p":
            raise ValueError(f"Expected CFR yuv420p source at {fps} FPS: {source}")
        origin = (video.start_time or 0) * video.time_base
        stream = out.add_stream("libx264", rate=rate, options={"crf": "0", "preset": "ultrafast"})
        stream.width, stream.height = video.width, video.height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 2
        stream.time_base = 1 / rate
        inp.seek(int((origin + Fraction(start, 1) / rate) / video.time_base), stream=video, backward=True)
        for frame in inp.decode(video=0):
            if frame.pts is None:
                raise ValueError("Source frame has no PTS")
            position = (frame.pts * frame.time_base - origin) * rate
            index = round(position)
            if abs(position - index) > 0.01:
                raise ValueError("Nonuniform source video timestamps")
            if index < start:
                continue
            if index >= start + count:
                break
            if index != start + written:
                raise ValueError(f"Missing/duplicate source frame: {source}, {index}")
            digest.update(frame.to_ndarray(format="yuv420p").tobytes())
            frame.pts, frame.time_base = written, 1 / rate
            for packet in stream.encode(frame):
                out.mux(packet)
            written += 1
        for packet in stream.encode():
            out.mux(packet)
    if written != count:
        raise ValueError(f"Truncated source clip: {source}, wanted {count}, got {written}")
    check, decoded = hashlib.sha256(), 0
    with av.open(str(target)) as inp:
        for frame in inp.decode(video=0):
            if frame.pts is None or abs(frame.pts * frame.time_base - decoded / rate) > Fraction(1, 10000):
                raise ValueError(f"Output timestamp mismatch: {target}")
            check.update(frame.to_ndarray(format="yuv420p").tobytes())
            decoded += 1
    if decoded != count or check.digest() != digest.digest():
        raise ValueError(f"Output frame count/pixel mismatch: {target}")
    return dict(video_size_bytes=target.stat().st_size, video_sha256=sha256(target),
                decoded_yuv_sha256=check.hexdigest(), video_processing="frame_exact_lossless_h264")


def convert(plan, manifest, output, limit=None):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "export-status.json", {"status": "incomplete"})
    write_json(output / "source-manifest.json", manifest)
    hashes, stats, mapping, split_tasks = {}, [], [], []
    for item in plan:
        task = item["task"]
        folder = f"{task['task_id'][1:]}-{task['task_slug']}"
        target = output / "Order" / folder
        target.mkdir(parents=True)
        (target / "annotations").mkdir()
        episodes = item["episodes"][:limit] if limit else item["episodes"]
        rows = []
        for index, episode in enumerate(episodes):
            views = {}
            for view, source in episode["views"].items():
                path = target / "videos" / view / f"episode_{index:06d}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = source["path"]
                if raw not in hashes:
                    hashes[raw] = (sha256(raw), raw.stat())
                details = export_clip(raw, path, source["start"], episode["frames"], episode["fps"])
                views[view] = dict(camera=CAMERAS[view], video_path=path.relative_to(target).as_posix(),
                                   source_path=source["source_path"], source_sha256=hashes[raw][0],
                                   **{k: v for k, v in details.items() if k != "decoded_yuv_sha256"})
                mapping.append(dict(sample_id=f"evomind_{task['task_id']}_episode_{index:06d}",
                                    source_dataset=episode["source_dataset"], source_episode=episode["source_episode"],
                                    view=view, source_start_frame=source["start"], frames=episode["frames"], **details))
            main = views["main_view"]
            row = dict(sample_id=f"evomind_{task['task_id']}_episode_{index:06d}",
                       task_id=task["task_id"], task_name=task["task_name"], task_slug=task["task_slug"],
                       domain="real-world", episode_index=index, video_path=main["video_path"],
                       annotation_path=None, has_annotation=False, task_prompt=None,
                       source_instruction=episode["instruction"], fps=episode["fps"], total_frames=episode["frames"],
                       num_stages=None, annotation_sha256=None, video_size_bytes=main["video_size_bytes"],
                       video_sha256=main["video_sha256"], source_path=main["source_path"],
                       source_total_frames=episode["frames"], source_sha256=main["source_sha256"],
                       video_processing=main["video_processing"], source_repo=None, source_revision=None,
                       video_views=views, suite="Order", source_dataset="EvoMind-PiperX",
                       source_sample_id=f"{episode['source_dataset']}/episode_{episode['source_episode']:06d}",
                       source_dataset_video_path=main["source_path"], collection_task_id=task["collection_task_id"],
                       collection_task_name=task["collection_task_name"], collection_task_slug=task["task_slug"],
                       source_collection="EvoMind", source_dataset_url=None,
                       source_task_id=task["collection_task_id"], source_task_name=episode["instruction"],
                       source_task_url=None, name_evidence="recorded_lerobot_episode_instruction", source_file_url=None)
            rows.append(row)
            print(f"{task['task_id']} episode {index + 1}/{len(episodes)} verified", flush=True)
        for path, digest in item["snapshots"].items():
            if sha256(path) != digest:
                raise ValueError(f"Source metadata changed during export: {path}")
        summary = dict(task_id=task["task_id"], task_name=task["task_name"], path=".",
                       task_slug=task["task_slug"], suite="Order", domain="real-world",
                       domain_counts={"real-world": len(rows)}, source_dataset="EvoMind-PiperX",
                       source_repositories=[], source_revisions=[], source_collection="EvoMind",
                       collection_task_id=task["collection_task_id"], collection_task_names=[task["collection_task_name"]],
                       trajectory_count=len(rows), annotated_trajectory_count=0, unannotated_trajectory_count=len(rows),
                       view_trajectory_counts={view: len(rows) for view in CAMERAS},
                       video_file_count=len(rows) * 3, annotation_file_count=0, trajectory_index="metadata.jsonl",
                       classification_basis="User classified these recordings as Order tasks.",
                       format_reference={"repository": manifest["format_reference"], "revision": manifest["format_revision"]},
                       source_datasets=[source["dataset"] for source in task["sources"]],
                       annotation_status="No stage labels collected; pending annotation.")
        write_json(target / "metadata.json", summary)
        index_path = target / "metadata.jsonl"
        index_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        validation_count = math.ceil(len(rows) * 0.2)
        sample_ids = [row["sample_id"] for row in rows]
        split_tasks.append(dict(suite="Order", task_id=task["task_id"], task_name=task["task_name"],
                                task_path=f"Order/{folder}", trajectory_index=f"Order/{folder}/metadata.jsonl",
                                trajectory_index_sha256=sha256(index_path),
                                train=sample_ids[:-validation_count], val=sample_ids[-validation_count:]))
        stats.append(dict(task_id=task["task_id"], directory=f"Order/{folder}", episodes=len(rows),
                          frames=sum(e["frames"] for e in episodes), warnings=item["warnings"]))
    for path, (digest, before) in hashes.items():
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError(f"Source video changed during export: {path}")
    write_json(output / "provenance.json", dict(tool_sha256=sha256(__file__), trajectories=mapping,
                                               source_metadata_sha256={p: h for t in plan for p, h in t["snapshots"].items()}))
    write_json(output / "manifest.additions.json", dict(
        schema_version=1, partial=bool(limit), base_revision=manifest["format_revision"],
        note="New tasks only; merge explicitly into the HF manifest without changing existing frozen splits.",
        split_policy={"unit": "trajectory", "val_fraction": 0.2,
                      "selection": "last ceil(n * 0.2) trajectories are val; preceding trajectories are train",
                      "include_unannotated": True, "path_base": "manifest directory", "frozen": True},
        tasks=split_tasks))
    report = dict(status="passed", partial=bool(limit), tasks=stats,
                  episodes=sum(t["episodes"] for t in stats), videos=sum(t["episodes"] * 3 for t in stats),
                  checks="Full output decode; frame counts, timestamps and decoded YUV equality to source cuts; source metadata hashes.")
    write_json(output / "export-status.json", report)
    (output / "Order/README.md").write_text(
        "# Order — EvoMind recordings\n\nUnannotated real robot recordings. "
        "Each task has metadata.json and metadata.jsonl with task-relative paths. "
        "main_view is the environment camera; left_view/right_view are wrist cameras. "
        "Stage annotations have not been collected. See ../export-status.json and ../provenance.json.\n\n"
        + "\n".join(f"- [{t['task_id']}]({t['directory'].split('/', 1)[1]}/): {t['episodes']} trajectories" for t in stats) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path(__file__).with_name("order_20261031.json"))
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--plan", action="store_true", help="Validate metadata and report short clips without writing output")
    p.add_argument("--limit-per-task", type=int, help="Export a partial smoke test, never a complete release")
    args = p.parse_args()
    if args.limit_per_task is not None and args.limit_per_task < 1:
        p.error("--limit-per-task must be positive")
    if not args.plan and args.output is None:
        p.error("--output is required unless --plan is set")
    manifest = json.loads(args.manifest.read_text())
    plan = load_plan(manifest, args.source_root.resolve())
    if args.plan:
        report = dict(episodes=sum(len(t["episodes"]) for t in plan), tasks=[
            dict(task_id=t["task"]["task_id"], name=t["task"]["collection_task_name"],
                 episodes=len(t["episodes"]), warnings=t["warnings"]) for t in plan])
    else:
        output = args.output.resolve()
        if output.is_relative_to(args.source_root.resolve()) or args.source_root.resolve().is_relative_to(output):
            p.error("Output must be separate from the source data tree")
        report = convert(plan, manifest, output, args.limit_per_task)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
