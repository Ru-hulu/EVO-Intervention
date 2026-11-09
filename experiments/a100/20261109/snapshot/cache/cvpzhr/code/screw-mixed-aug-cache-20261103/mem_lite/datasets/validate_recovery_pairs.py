"""Audit paired caches before publishing disjoint train/validation/review lists."""

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path

import torch

from mem_lite.datasets.m3_dataset import DATA_FORMAT, M3Dataset, VIEW_ORDER
from mem_lite.datasets.prepare_recovery import sha256, write_json
from mem_lite.datasets.prepare_recovery_pairs import read_records


def validate(prepared_dir, cache_dir, reference_config, workers=4):
    torch.set_num_threads(1)
    prepared, cache = Path(prepared_dir), Path(cache_dir)
    rows = read_records(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    reference = json.loads(Path(reference_config).read_text())
    clips = json.loads((prepared / "clips-report.json").read_text())
    expected_clips = {p for row in rows for p in row["videos"]}
    if {c["path"] for c in clips["clips"]} != expected_clips or clips["num_clips"] != len(rows) * 3:
        raise ValueError("Missing synchronized paired clips")
    groups = {split: set(manifest["splits"][split]) for split in ("train", "validation", "review")}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "review"), ("validation", "review"))):
        raise ValueError("Source-video leakage between splits")
    baseline_files = reference.get("sample_files", [])
    if isinstance(baseline_files, str):
        baseline_files = json.loads(Path(baseline_files).read_text())
    baseline_videos = set()
    for path in baseline_files:
        path = Path(path)
        directory = path if path.is_dir() else path.parent
        source = json.loads((directory / "summary.json").read_text()).get("source", {})
        if source.get("task") == "insert-screw-recover":
            baseline_videos.add(source["video_id"])
    overlap = baseline_videos & groups["validation"]
    if overlap:
        raise ValueError(f"Validation videos were used to train the initialization checkpoint: {sorted(overlap)}")

    def audit(row):
        directory = cache / row["split"] / row["episode_id"]
        summary_path, samples_path = directory / "summary.json", directory / "samples.jsonl"
        summary = json.loads(summary_path.read_text())
        if (summary["format_version"] != DATA_FORMAT or summary["source"] != row
                or summary["task_prompt"] != row["task_prompt"]
                or summary["history_source"] != reference["history_source"]
                or summary["history_dim"] != reference["model"]["history_dim"]
                or summary["history_capacity"] != reference["history_capacity"]):
            raise ValueError(f"Cache metadata mismatch: {directory}")
        if row["source_video_id"] not in groups[row["split"]]:
            raise ValueError("Pair assigned to the wrong source-video split")
        if (row["label_status"] == "pending_review") != (row["split"] == "review"):
            raise ValueError("Unreviewed pair entered train/validation")
        with (prepared / row["progress_csv"]).open() as handle:
            labels = {int(r["frame_index"]): float(r["pt"]) for r in csv.DictReader(handle)}
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
        # The source and all clips are exactly 30 FPS; cached observations are 5 FPS.
        if row["fps"] != 30.0 or reference["history_source"]["sample_fps"] != 5.0:
            raise ValueError("This audit expects 30 FPS source / 5 FPS observations")
        last = row["frame_range"][1] - 1
        expected = sorted(set(range(0, last + 1, 6)) | {last})
        if [s["frame_index"] for s in samples] != expected or len(samples) != summary["num_samples"]:
            raise ValueError(f"Missing or duplicated observation: {directory}")
        previous, seam_checked, history_count, max_tokens = None, False, 0, 0
        for sample in samples:
            frame = sample["frame_index"]
            if sample["pt"] != labels[frame] or (sample["history"] is None) != (previous is None):
                raise ValueError(f"Incorrect PT or memory reset at {directory}:{frame}")
            for view in VIEW_ORDER:
                image = sample["images"][view]
                if image is None or (directory / image).stat().st_size == 0:
                    raise ValueError(f"Missing camera {view}: {directory}:{frame}")
            if sample["history"] is not None:
                saved = torch.load(directory / sample["history"], map_location="cpu", weights_only=True)
                tokens = saved["tokens"]
                if (saved["through_frame"] != previous or tokens.ndim != 2
                        or tokens.shape[1] != summary["history_dim"]
                        or not 0 < len(tokens) <= summary["history_capacity"]
                        or not torch.isfinite(tokens).all()):
                    raise ValueError(f"Invalid or non-causal history: {directory}:{frame}")
                history_count += 1
                max_tokens = max(max_tokens, len(tokens))
            if not seam_checked and frame >= row["second_round_start"]:
                if previous is None or sample["history"] is None:
                    raise ValueError(f"Memory lost at the internal round boundary: {directory}")
                seam_checked = True
            previous = frame
        if not seam_checked or samples[-1]["pt"] != 1.0:
            raise ValueError(f"Incomplete paired task: {directory}")
        return {"episode_id": row["episode_id"], "split": row["split"], "samples": len(samples),
                "histories": history_count, "max_history_tokens": max_tokens,
                "sample_file": str(samples_path), "samples_sha256": sha256(samples_path),
                "summary_sha256": sha256(summary_path), "internal_boundary_memory_checked": True}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(audit, rows):
            results.append(result)
            if len(results) % 8 == 0:
                print(json.dumps({"audited_pairs": len(results), "total_pairs": len(rows)}), flush=True)
    splits = {}
    for split in groups:
        files = [r["sample_file"] for r in results if r["split"] == split]
        dataset = M3Dataset(files, history_dim=reference["model"]["history_dim"])
        # Decode real images and load history through the training dataset interface.
        for index in (0, min(1, len(dataset) - 1), len(dataset) - 1):
            dataset[index]
        splits[split] = {"episodes": len(files), "observations": len(dataset), "signature": dataset.signature}
        write_json(cache / f"{split}.sample_files.json", files)
    report = {"status": "verified", "pairs": len(rows), "clips": clips["num_clips"],
              "splits": splits, "observations": sum(r["samples"] for r in results),
              "history_tensors_checked": sum(r["histories"] for r in results),
              "all_cached_pt_matches_dense_labels": True, "all_histories_finite_and_causal": True,
              "source_video_leakage": False, "review_excluded": True,
              "baseline_recovery_training_videos": sorted(baseline_videos),
              "baseline_training_overlap_with_validation": sorted(overlap),
              "memory_continuous_across_internal_boundary": True,
              "prepared_dir": str(prepared), "cache_dir": str(cache), "episodes": results}
    write_json(cache / "validation-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--workers", type=int, default=4)
    report = validate(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
