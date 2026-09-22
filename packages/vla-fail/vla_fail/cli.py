"""Fit, calibrate, and replay saved episodes without loading a VLA model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .acc import ACCDetector
from .calibration import calibrate_threshold
from .detector import VLAFAILDetector
from .llmd import LLMDDetector


def load_episode(path):
    with np.load(path, allow_pickle=False) as data:
        episode = {key: data[key].copy() for key in data.files}
    episode["metadata"] = json.loads(str(episode["metadata"].item()))
    episode["episode_id"] = str(episode["episode_id"].item())
    if not episode["episode_id"]:
        raise ValueError("episode_id must not be empty")
    return episode


def validate_reference(episode, detector):
    for key, value in detector.metadata.items():
        if key != "fit_episode_ids" and episode["metadata"].get(key) != value:
            raise ValueError(f"Feature metadata mismatch: {key}")
    if detector.fixed_noise is None or not np.array_equal(episode["fixed_noise"], detector.fixed_noise):
        raise ValueError("All episodes must use the saved fixed_noise")


def fit(paths, output, regularization=1e-6):
    detector = None
    ids = []
    for path in paths:
        episode = load_episode(path)
        if episode["episode_id"] in ids:
            raise ValueError("Duplicate training episode_id")
        if detector is None:
            detector = LLMDDetector(regularization, fixed_noise=episode["fixed_noise"], metadata=episode["metadata"])
        validate_reference(episode, detector)
        detector.partial_fit(episode["features"])
        ids.append(episode["episode_id"])
    if detector is None:
        raise ValueError("At least one training episode is required")
    detector.metadata["fit_episode_ids"] = ids
    detector.finalize().save(output)
    return {"samples": detector.n_samples, "feature_shape": detector.feature_shape, "episodes": len(ids)}


def replay(episode, *, llmd=None, acc_config=None):
    if llmd is not None:
        validate_reference(episode, llmd)
    acc = None
    if acc_config is not None:
        acc = ACCDetector(
            alpha=acc_config.get("ema_alpha", 0.9),
            v_min=acc_config.get("v_min", 0.001),
            threshold=acc_config.get("threshold"),
        )
    detector = VLAFAILDetector(llmd=llmd, acc=acc)
    count = len(episode["features"] if llmd is not None else episode["positions"])
    if count == 0:
        raise ValueError("Episode has no observations")
    if acc is not None and (len(episode["positions"]) != count or episode["executed_steps"].shape != (count,)):
        raise ValueError("positions and executed_steps must align with feature observations")
    rows = []
    for index in range(count):
        result = detector.step(
            tokens=None if llmd is None else episode["features"][index],
            positions=None if acc is None else episode["positions"][index],
            executed_steps=None if acc is None or index == 0 else episode["executed_steps"][index],
        )
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "step": index,
                "llmd": None if result.llmd is None else result.llmd.score,
                "acc": None if result.acc is None else result.acc.score,
                "acc_raw": None if result.acc is None else result.acc.raw_score,
                "need_intervention": result.need_intervention,
                "decision_complete": result.decision_complete,
            }
        )
    return rows


def calibrate(paths, output_dir, *, stats=None, acc_config=None, alpha=0.05):
    llmd = None if stats is None else LLMDDetector.load(stats)
    if llmd is None and acc_config is None:
        raise ValueError("Enable --stats and/or --acc-config")
    signals = (["llmd"] if llmd is not None else []) + (["acc"] if acc_config is not None else [])
    scores = {signal: [] for signal in signals}
    ids = set()
    for path in paths:
        episode = load_episode(path)
        success = episode.get("success")
        if success is None or success.shape != () or success.dtype.kind != "b" or not success.item():
            raise ValueError("Calibration requires explicitly labeled successful episodes")
        eid = episode["episode_id"]
        if eid in ids or (llmd is not None and eid in llmd.metadata.get("fit_episode_ids", [])):
            raise ValueError("Calibration episodes must be unique and disjoint from training")
        ids.add(eid)
        rows = replay(episode, llmd=llmd, acc_config=acc_config)
        for signal in signals:
            scores[signal].append([row[signal] for row in rows if row[signal] is not None])
    # Union bound allocates the requested episode false-alarm budget across OR signals.
    reports = {signal: calibrate_threshold(scores[signal], alpha / len(signals)).to_dict() for signal in signals}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if llmd is not None:
        llmd.threshold = reports["llmd"]["threshold"]
        llmd.save(destination / "llmd.npz")
    if acc_config is not None:
        config = {**acc_config, "threshold": reports["acc"]["threshold"]}
        (destination / "acc.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"alpha_total": alpha, "calibration_episode_ids": sorted(ids), "signals": reports}
    (destination / "calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument("episodes", nargs="+")
    fitting.add_argument("--output", required=True)
    fitting.add_argument("--regularization", type=float, default=1e-6)
    for command in ("calibrate", "score"):
        sub = commands.add_parser(command)
        sub.add_argument("episodes", nargs="+")
        sub.add_argument("--stats")
        sub.add_argument("--acc-config", help="JSON; replay positions must already be absolute Cartesian XYZ")
        sub.add_argument("--output", required=True)
        if command == "calibrate":
            sub.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "fit":
        print(json.dumps(fit(args.episodes, args.output, args.regularization)))
        return
    acc_config = None if args.acc_config is None else json.loads(Path(args.acc_config).read_text(encoding="utf-8"))
    if args.command == "calibrate":
        report = calibrate(args.episodes, args.output, stats=args.stats, acc_config=acc_config, alpha=args.alpha)
        print(json.dumps({"episodes": len(report["calibration_episode_ids"]), "alpha_total": args.alpha}))
        return
    llmd = None if args.stats is None else LLMDDetector.load(args.stats)
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for path in args.episodes:
            rows = replay(load_episode(path), llmd=llmd, acc_config=acc_config)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
