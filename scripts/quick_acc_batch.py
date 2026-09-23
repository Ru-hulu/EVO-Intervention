"""Small ACC batch: reuse compatible predictions, then select extremes per group.

Score on the GPU host; render on the host holding the original camera videos.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

from batch_acc_videos import atomic_json
from batch_acc_videos import predict_one
from batch_acc_videos import render_one
from batch_acc_videos import report_one
import numpy as np


def ordered_records(plan):
    return sorted(plan["records"], key=lambda r: ("reuse" not in r, r["source"]["name"], r["index"]))


def choose_groups(records):
    groups = {}
    for record in records:
        groups.setdefault(record["source"], []).append(record)
    result = {}
    for group, values in groups.items():
        if len(values) < 2:
            raise ValueError("Each group needs at least two episodes")
        ordered = sorted(values, key=lambda r: (r["acc_p95"], r["acc_median"], r["episode_id"]))
        result[group] = {"lowest": ordered[0], "highest": ordered[-1], "ranked": ordered}
    return result


def score(plan, output):
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
    from replay_lerobot_vla_fail import load_policy
    import torch

    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    atomic_json(
        output / "run_manifest.json",
        {"plan": plan, "fingerprint": fingerprint, "mode": "ACC only; subset, not full-dataset ranking"},
    )
    records = ordered_records(plan)
    model = None
    statistics = []
    for index, record in enumerate(records):
        atomic_json(
            output / "score_status.json",
            {"phase": "scoring", "completed": index, "total": len(records), "episode": record["episode_id"]},
        )
        destination = output / "predictions" / (record["episode_id"] + ".npz")
        if not destination.exists() and "reuse" in record:
            with np.load(record["reuse"], allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata["checkpoint"] != plan["checkpoint"] or metadata["dtype"] != "float32":
                    raise ValueError("Cannot reuse another model's predictions")
                frames = archive["frame_indices"]
                if not np.array_equal(frames, np.arange(0, record["length"], plan["stride"])):
                    raise ValueError("Cached predictions do not cover the selected timeline")
                provenance = json.loads(str(archive["provenance"].item()))
                if (
                    provenance["dataset_root"] != record["source"]["root"]
                    or provenance["episode_index"] != record["index"]
                ):
                    raise ValueError("Cached prediction source does not match")
                provenance.update(
                    {
                        "label": record["source"]["label"],
                        "sample_stride": plan["stride"],
                        "reused_predictions": record["reuse"],
                    }
                )
                arrays = {
                    name: archive[name]
                    for name in [
                        "positions_left",
                        "positions_right",
                        "predicted_actions",
                        "frame_indices",
                        "timestamps",
                        "query_latency_s",
                    ]
                }
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.stem + ".partial.npz")
                np.savez_compressed(
                    temporary, **arrays, provenance=np.array(json.dumps(provenance)), fingerprint=np.array(fingerprint)
                )
                temporary.replace(destination)
        if not destination.exists():
            bundle = output / "scratch" / record["source"]["name"] / f"episode_{record['index']:06d}.npz"
            deadline = time.monotonic() + 2 * 3600
            while not bundle.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Missing observation bundle: {bundle}")
                time.sleep(5)
            if model is None:
                torch.set_num_threads(4)
                model, (preprocess, postprocess) = load_policy(
                    Path(plan["checkpoint"]), "cuda", Path(plan["tokenizer"])
                )
                fk = C_PiperForwardKinematics(dh_is_offset=1)
                print("WEIGHTS_LOADED_STRICTLY", flush=True)
            destination = predict_one(record, plan, model, preprocess, postprocess, fk, fingerprint, output)
        statistics.append(report_one(record, plan, destination, fingerprint, output))
        print("SCORED", record["episode_id"], flush=True)
    if model is not None:
        del model, preprocess, postprocess
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(output / "selection.json", choose_groups(statistics))
    atomic_json(output / "SCORING_COMPLETE.json", {"episodes": len(statistics), "statistics": statistics})
    atomic_json(output / "score_status.json", {"phase": "complete", "completed": len(statistics), "total": len(records)})


def render_remote(plan, output, peer):
    records = ordered_records(plan)
    statistics = []
    options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for index, record in enumerate(records):
            atomic_json(
                output / "render_status.json",
                {
                    "phase": "waiting_or_rendering",
                    "reports_received": index,
                    "total": len(records),
                    "episode": record["episode_id"],
                },
            )
            report = output / "reports" / record["episode_id"]
            report.mkdir(parents=True, exist_ok=True)
            remote_summary = str(report / "summary.json")
            deadline = time.monotonic() + 2 * 3600
            while subprocess.run(["ssh", "-n", *options, peer, "test -f " + remote_summary], check=False).returncode:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Scoring did not finish: {record['episode_id']}")
                time.sleep(5)
            for name in ["scores.csv", "summary.json"]:
                subprocess.run(["scp", *options, peer + ":" + str(report / name), str(report / name)], check=True)
            summary = json.loads((report / "summary.json").read_text())
            metrics = {key: summary[key] for key in ["acc_p95", "acc_median", "acc_max", "duration_s"]}
            statistics.append(
                {
                    "episode_id": record["episode_id"],
                    "source": record["source"]["name"],
                    "report": str(report),
                    **metrics,
                }
            )
            name = (
                "成功组示范" if record["source"]["name"] == "hf_success" else "失败组"
            ) + f"_{record['index']:06d}_ACC"
            futures.append(executor.submit(render_one, report, output / "videos" / f"{name}.mp4", title=name))
        for future in futures:
            print("VIDEO_READY", future.result(), flush=True)
    selection = choose_groups(statistics)
    atomic_json(output / "selection.json", selection)
    with (output / "ranking.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_rank", *statistics[0].keys()])
        writer.writeheader()
        for group in selection.values():
            for rank, record in enumerate(group["ranked"], 1):
                writer.writerow({"group_rank": rank, **record})
    # Re-render just the four selected episodes on common axes for direct comparison.
    chosen = [
        (group, extreme, values[extreme]) for group, values in selection.items() for extreme in ("lowest", "highest")
    ]
    limits = {"acc_raw": 0.01, "acc": 0.01, "mae_mm": 0.01}
    for _, _, record in chosen:
        with (Path(record["report"]) / "scores.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for key, fields in {
                    "acc_raw": ["acc_raw_left", "acc_raw_right"],
                    "acc": ["acc_left", "acc_right"],
                    "mae_mm": ["mae_mm_left", "mae_mm_right"],
                }.items():
                    limits[key] = max([limits[key], *(float(row[f]) * 1.05 for f in fields if row[f])])
    atomic_json(output / "selected_axis_limits.json", limits)
    selected_files = []
    for index, (group, extreme, record) in enumerate(chosen, 1):
        label = ("成功组" if group == "hf_success" else "失败组") + ("_ACC最低" if extreme == "lowest" else "_ACC最高")
        name = f"{index:02d}_{label}_{record['episode_id'].rsplit('_', 1)[-1]}"
        selected_files.append(
            render_one(
                record["report"],
                output / "selected" / f"{name}.mp4",
                title=name,
                limits=output / "selected_axis_limits.json",
            )
        )
    atomic_json(
        output / "COMPLETE.json",
        {
            "episodes": len(records),
            "selected": selected_files,
            "selection": selection,
            "completed_at_unix": time.time(),
        },
    )
    atomic_json(output / "render_status.json", {"phase": "complete", "episodes": len(records)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["score", "render"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer", default="zhaobo@192.168.105.96")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    if args.mode == "score":
        score(plan, args.output)
    else:
        render_remote(plan, args.output, args.peer)


if __name__ == "__main__":
    main()
