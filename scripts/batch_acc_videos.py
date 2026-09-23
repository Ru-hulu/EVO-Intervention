"""Resumable offline ACC replay, rank all episodes, and render complete videos.

One GPU model is loaded for the entire collection. No LLMD or robot connection.
The plan fixes dataset versions, camera mapping, checkpoint, seed and stride.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
from vla_fail.acc_batch import describe_rows
from vla_fail.acc_batch import rank_records
from vla_fail.acc_batch import score_episode


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def inventory(plan):
    records = []
    for source in plan["datasets"]:
        root = Path(source["root"])
        info = json.loads((root / "meta/info.json").read_text())
        if info["fps"] != plan["fps"]:
            raise ValueError("All datasets must use the declared action timebase")
        metadata = []
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            columns = [name for name in pq.read_schema(path).names if not name.startswith("stats/")]
            metadata.extend(pq.read_table(path, columns=columns).to_pylist())
        if (
            len(metadata) != source["expected_episodes"]
            or sum(m["length"] for m in metadata) != source["expected_frames"]
        ):
            raise ValueError(f"Dataset inventory mismatch: {source['name']}")
        ids = set()
        for meta in sorted(metadata, key=lambda m: m["episode_index"]):
            index = int(meta["episode_index"])
            if index in ids or meta["length"] <= plan["stride"]:
                raise ValueError("Duplicate or too-short episode")
            ids.add(index)
            # Validate all inputs before any ranking can be declared complete.
            for key in source["camera_map"].values():
                prefix = f"videos/{key}/"
                video = root / info["video_path"].format(
                    video_key=key, chunk_index=meta[prefix + "chunk_index"], file_index=meta[prefix + "file_index"]
                )
                if not video.is_file():
                    raise FileNotFoundError(video)
            records.append(
                {
                    "source": source,
                    "index": index,
                    "length": int(meta["length"]),
                    "episode_id": f"{source['name']}/episode_{index:06d}",
                }
            )
    return records


def predict_one(record, plan, policy, preprocess, postprocess, fk, fingerprint, output):
    from prepare_lerobot_vla_fail import prepare
    from replay_lerobot_vla_fail import xyz_from_joint_degrees
    import torch

    source, index = record["source"], record["index"]
    cache = output / "scratch" / source["name"]
    bundle_path = cache / f"episode_{index:06d}.npz"
    if not bundle_path.exists():
        prepare(
            source["root"],
            index,
            cache,
            stride=plan["stride"],
            base_camera=source["camera_map"]["base"],
            left_wrist_camera=source["camera_map"]["left_wrist"],
            right_wrist_camera=source["camera_map"]["right_wrist"],
        )
    with np.load(bundle_path, allow_pickle=False) as archive:
        bundle = {key: archive[key] for key in archive.files}
    provenance = json.loads(str(bundle["provenance"].item()))
    if provenance["dataset_root"] != source["root"] or provenance["sample_stride"] != plan["stride"]:
        raise ValueError("Stale input cache has different preprocessing/source")
    if len(bundle["frame_indices"]) != (record["length"] + plan["stride"] - 1) // plan["stride"]:
        raise ValueError("Incomplete observation coverage")
    provenance.update(
        {"source_origin": source["origin"], "dataset_version": source.get("revision"), "label": source["label"]}
    )
    names = {
        "base": "observation.images.base_0_rgb",
        "left_wrist": "observation.images.left_wrist_0_rgb",
        "right_wrist": "observation.images.right_wrist_0_rgb",
    }
    action_rows, left, right, latencies = [], [], [], []
    policy.reset()
    for i, frame in enumerate(bundle["frame_indices"]):
        observation = {"observation.state": torch.from_numpy(bundle["state"][i]), "task": str(bundle["task"].item())}
        for camera, key in names.items():
            observation[key] = torch.from_numpy(bundle[f"images_{camera}"][i]).permute(2, 0, 1)
        batch = preprocess(observation)
        generator = torch.Generator(device="cuda").manual_seed(plan["action_seed"] + int(frame))
        noise = torch.randn(
            (1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator, device="cuda"
        )
        started = time.monotonic()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch, noise=noise)
            actions = postprocess(actions)[0].detach().float().cpu().numpy()
        latencies.append(time.monotonic() - started)
        xyz = xyz_from_joint_degrees(actions, fk)
        action_rows.append(actions)
        left.append(xyz["positions_left"])
        right.append(xyz["positions_right"])
    prediction = output / "predictions" / source["name"] / f"episode_{index:06d}.npz"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    temporary = prediction.with_name(prediction.stem + ".partial.npz")
    np.savez_compressed(
        temporary,
        positions_left=np.stack(left),
        positions_right=np.stack(right),
        predicted_actions=np.stack(action_rows),
        frame_indices=bundle["frame_indices"],
        timestamps=bundle["timestamps"],
        provenance=np.array(json.dumps(provenance)),
        fingerprint=np.array(fingerprint),
        query_latency_s=np.asarray(latencies),
    )
    temporary.replace(prediction)
    # Only the exact disposable input cache created by this runner is removed.
    if not bundle_path.resolve().is_relative_to((output / "scratch").resolve()):
        raise ValueError("Unexpected scratch path")
    bundle_path.unlink()
    print(
        json.dumps(
            {"predicted": record["episode_id"], "queries": len(latencies), "mean_model_s": float(np.mean(latencies))}
        ),
        flush=True,
    )
    return prediction


def report_one(record, plan, prediction, fingerprint, output):
    with np.load(prediction, allow_pickle=False) as archive:
        if str(archive["fingerprint"].item()) != fingerprint:
            raise ValueError("Prediction belongs to a different plan/checkpoint")
        rows = score_episode(
            record["episode_id"],
            archive["positions_left"],
            archive["positions_right"],
            archive["frame_indices"],
            archive["timestamps"],
            alpha=plan["alpha"],
            v_min=plan["v_min"],
        )
        provenance = json.loads(str(archive["provenance"].item()))
        inference_s = float(archive["query_latency_s"].sum())
    metrics = describe_rows(rows)
    report = output / "reports" / record["episode_id"]
    report.mkdir(parents=True, exist_ok=True)
    with (report / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "metric_mode": "acc",
        "episode_id": record["episode_id"],
        "fingerprint": fingerprint,
        "provenance": provenance,
        "feature_metadata": {"checkpoint": plan["checkpoint"], "dtype": "float32"},
        "acc_threshold": None,
        "ema_alpha": plan["alpha"],
        "v_min_m": plan["v_min"],
        "observations": len(rows),
        "duration_s": record["length"] / plan["fps"],
        "model_inference_s": inference_s,
        "query_interval_s": plan["stride"] / plan["fps"],
        "action_horizon": 50,
        "overlap_steps": 50 - plan["stride"],
        "evaluation_notes": "Offline counterfactual replay of recorded observations, RTC disabled. Predicted chunks, not logged runtime chunks. ACC is consistency, not task-success probability. Zero-initialized EMA; first observation has no ACC. No calibrated threshold.",
        **metrics,
    }
    atomic_json(report / "summary.json", summary)
    return {
        "episode_id": record["episode_id"],
        "report": str(report),
        "source": record["source"]["name"],
        "label": record["source"]["label"],
        "duration_s": summary["duration_s"],
        **metrics,
    }


def render_one(report, destination, *, title=None, limits=None):
    destination = Path(destination)
    marker = destination.with_suffix(".json")
    if destination.exists() and marker.exists():
        saved = json.loads(marker.read_text())
        if saved["report"] == str(Path(report).resolve()) and saved["frames"] > 0:
            return str(destination)
        raise ValueError("Mismatched existing video")
    if destination.exists():
        # Preserve an incomplete render for diagnosis before retrying.
        destination.replace(destination.with_name(destination.stem + f".interrupted_{time.time_ns()}.mp4"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).with_name("render_vla_fail_video.py")),
        "--report",
        str(report),
        "--output",
        str(destination),
    ]
    if title:
        command += ["--title", title]
    if limits:
        command += ["--axis-limits", str(limits)]
    with destination.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return str(destination)


def run(plan, output, *, max_episodes=None):
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
    from replay_lerobot_vla_fail import load_policy
    import torch

    started = time.time()

    def status(phase, **details):
        atomic_json(
            output / "status.json",
            {"phase": phase, "updated_at_unix": time.time(), "elapsed_s": time.time() - started, **details},
        )

    try:
        if plan.get("stage_marker"):
            status("waiting_for_staged_inputs")
            deadline = time.monotonic() + 48 * 3600
            while not Path(plan["stage_marker"]).exists():
                failure = Path(plan["stage_marker"]).with_name("STAGE_FAILED.txt")
                if failure.exists():
                    raise RuntimeError(failure.read_text())
                if time.monotonic() > deadline:
                    raise TimeoutError("Input staging did not finish in 48 hours")
                time.sleep(30)
        records = inventory(plan)
        if max_episodes:
            records = records[:max_episodes]
        stage = (
            json.loads(Path(plan["stage_marker"]).read_text())
            if plan.get("stage_marker")
            else {"checkpoint": plan["checkpoint"]}
        )
        fingerprint = hashlib.sha256(json.dumps({"plan": plan, "stage": stage}, sort_keys=True).encode()).hexdigest()
        if (output / "run_manifest.json").exists():
            prior = json.loads((output / "run_manifest.json").read_text())
            if prior["fingerprint"] != fingerprint:
                raise ValueError("Cannot resume outputs from another plan")
        atomic_json(
            output / "run_manifest.json",
            {
                "plan": plan,
                "fingerprint": fingerprint,
                "episodes": len(records),
                "sampled_observations": sum((r["length"] + plan["stride"] - 1) // plan["stride"] for r in records),
                "source_frames": sum(r["length"] for r in records),
                "ranking": "ascending P95(max(left_EMA,right_EMA)); ties by median then episode_id; first missing ACC excluded",
                "stage": stage,
            },
        )
        pending = [r for r in records if not (output / "predictions" / (r["episode_id"] + ".npz")).exists()]
        if pending:
            status("waiting_for_gpu", total=len(records))
            deadline = time.monotonic() + 48 * 3600
            while True:
                free = int(
                    subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
                    .decode()
                    .splitlines()[0]
                )
                if free >= 21000:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("GPU did not become available in 48 hours")
                time.sleep(30)
            torch.set_num_threads(4)
            policy, (preprocess, postprocess) = load_policy(Path(plan["checkpoint"]), "cuda", Path(plan["tokenizer"]))
            if policy.config.chunk_size != 50 or not 0 < plan["stride"] < 50:
                raise ValueError("Expected H=50 and an overlapping query interval")
            fk = C_PiperForwardKinematics(dh_is_offset=1)
            print("WEIGHTS_LOADED_STRICTLY_ACC_ONLY", flush=True)
        statistics = []
        for completed, record in enumerate(records):
            status("inference", completed=completed, total=len(records), current=record["episode_id"])
            prediction = output / "predictions" / (record["episode_id"] + ".npz")
            if not prediction.exists():
                prediction = predict_one(record, plan, policy, preprocess, postprocess, fk, fingerprint, output)
            statistics.append(report_one(record, plan, prediction, fingerprint, output))
            gc.collect()
        if pending:
            del policy, preprocess, postprocess
            gc.collect()
            torch.cuda.empty_cache()
        if max_episodes:
            status("smoke_complete", completed=len(records))
            return
        ordered, selection = rank_records(statistics)
        atomic_json(output / "ranking.json", ordered)
        with (output / "ranking.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["rank", *ordered[0].keys()])
            writer.writeheader()
            for rank, row in enumerate(ordered, 1):
                writer.writerow({"rank": rank, **row})
        atomic_json(output / "selection.json", selection)
        chosen = [*selection["lowest"], *selection["highest"]]
        limits = {"acc_raw": 0.01, "acc": 0.01, "mae_mm": 0.01}
        for record in chosen:
            with (Path(record["report"]) / "scores.csv").open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for key, fields in {
                        "acc_raw": ["acc_raw_left", "acc_raw_right"],
                        "acc": ["acc_left", "acc_right"],
                        "mae_mm": ["mae_mm_left", "mae_mm_right"],
                    }.items():
                        limits[key] = max([limits[key], *(float(row[f]) * 1.05 for f in fields if row[f])])
        atomic_json(output / "selected_axis_limits.json", limits)
        for index, record in enumerate(chosen):
            status("render_selected", completed=index, total=4)
            category = "ACC最低" if index < 2 else "ACC最高"
            name = f"{index + 1:02d}_{category}_{record['episode_id'].replace('/', '_')}"
            render_one(
                record["report"],
                output / "selected" / f"{name}.mp4",
                title=name,
                limits=output / "selected_axis_limits.json",
            )
        atomic_json(output / "SELECTED_COMPLETE.json", {"selected": selection, "completed_at_unix": time.time()})
        # Encoding workers are subprocesses so matplotlib is never shared across threads.
        with ThreadPoolExecutor(max_workers=plan.get("render_workers", 2)) as executor:
            futures = [
                (r, executor.submit(render_one, r["report"], output / "videos" / (r["episode_id"] + ".mp4")))
                for r in records
            ]
            for index, (record, future) in enumerate(futures):
                future.result()
                status("render_all", completed=index + 1, total=len(records), current=record["episode_id"])
        atomic_json(
            output / "COMPLETE.json",
            {
                "episodes": len(records),
                "selected_videos": 4,
                "elapsed_s": time.time() - started,
                "completed_at_unix": time.time(),
            },
        )
        status("complete", completed=len(records), total=len(records))
    except Exception as exc:
        status("failed", error=repr(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, help="Smoke test only; skips final ranking and encoding")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # A persistent lock prevents an accidental second runner from duplicating GPU work.
    import fcntl

    with (args.output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(json.loads(args.plan.read_text(encoding="utf-8")), args.output.resolve(), max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
