"""Plot LLMD and per-arm ACC from real, exported model predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .acc import ACCDetector
from .cli import load_episode
from .cli import validate_reference
from .llmd import LLMDDetector


def evaluate(episode, llmd, *, ema_alpha=0.9, v_min=0.001, acc_threshold=None):
    validate_reference(episode, llmd)
    features = episode["features"]
    count = len(features)
    times = np.asarray(episode.get("timestamps", np.arange(count)), dtype=float)
    frames = np.asarray(episode.get("frame_indices", np.arange(count)))
    steps = np.asarray(episode["executed_steps"])
    if count < 2 or times.shape != (count,) or frames.shape != (count,) or steps.shape != (count,):
        raise ValueError("Need at least two aligned feature/timestamp/frame/stride observations")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    arms = [arm for arm in ("left", "right") if f"positions_{arm}" in episode]
    if not arms and "positions" in episode:
        arms = ["eef"]
    if not arms:
        raise ValueError("ACC needs predicted absolute XYZ chunks, not recorded executed actions")
    detectors = {arm: ACCDetector(ema_alpha, v_min, threshold=acc_threshold) for arm in arms}
    for arm in arms:
        positions = episode["positions" if arm == "eef" else f"positions_{arm}"]
        if positions.shape != (count, features.shape[1], 3):
            raise ValueError(f"positions_{arm} must be [T,H,3] aligned with features")
    token_scores = []
    rows = []
    for index, feature in enumerate(features):
        result = llmd.score(feature)
        token_scores.append(result.per_token_scores)
        row = {
            "episode_id": episode["episode_id"],
            "query_index": index,
            "frame_index": int(frames[index]),
            "time_s": float(times[index]),
            "llmd": result.score,
            "llmd_worst_token": result.worst_token,
            "llmd_alarm": result.triggered,
        }
        decisions = [result.triggered]
        acc_scores = []
        for arm, detector in detectors.items():
            position = episode["positions" if arm == "eef" else f"positions_{arm}"][index]
            acc = detector.update(position, None if index == 0 else steps[index])
            row[f"acc_{arm}"] = acc.score
            row[f"acc_raw_{arm}"] = acc.raw_score
            row[f"mae_mm_{arm}"] = None if not acc.ready else float(np.mean(acc.per_axis_mae)) * 1000
            decisions.append(acc.triggered)
            if acc.score is not None:
                acc_scores.append(acc.score)
        row["acc_max"] = max(acc_scores) if acc_scores else None
        row["need_intervention"] = True if True in decisions else None if None in decisions else False
        rows.append(row)
    return rows, np.asarray(token_scores), arms


def render(
    episode_path,
    stats_path,
    output,
    *,
    ema_alpha=0.9,
    v_min=0.001,
    acc_threshold=None,
    bundle_path=None,
    failure_time=None,
):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    episode_path, output = Path(episode_path), Path(output)
    episode = load_episode(episode_path)
    llmd = LLMDDetector.load(stats_path)
    rows, token_scores, arms = evaluate(episode, llmd, ema_alpha=ema_alpha, v_min=v_min, acc_threshold=acc_threshold)
    output.mkdir(parents=True, exist_ok=True)
    times = np.array([r["time_s"] for r in rows])
    llmd_values = np.array([r["llmd"] for r in rows])
    acc_values = np.array([np.nan if r["acc_max"] is None else r["acc_max"] for r in rows])
    peak_llmd, peak_acc = int(llmd_values.argmax()), int(np.nanargmax(acc_values))
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 11),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.15, 0.95, 1.15, 0.9]},
    )
    episode_name = episode["episode_id"].rsplit("/", 1)[-1]
    model_name = Path(episode["metadata"].get("checkpoint", "unspecified")).name
    outcome = "unknown" if "success" not in episode else "success" if episode["success"].item() else "failure"
    fig.suptitle(
        f"{episode_name} | LLMD + ACC | recorded outcome: {outcome}\n"
        f"Evaluated checkpoint: {model_name} (offline replay)",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].plot(times, llmd_values, color="#2359a4", linewidth=1.4, label="LLMD: max token squared distance")
    if llmd.threshold is not None:
        axes[0].axhline(llmd.threshold, color="#b42318", linestyle="--", label="Calibrated threshold")
    else:
        axes[0].text(
            0.99,
            0.95,
            "No calibrated LLMD threshold",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#765b19",
        )
    axes[0].scatter(times[peak_llmd], llmd_values[peak_llmd], color="#b42318", zorder=5)
    axes[0].set_ylabel("Squared distance")
    axes[0].legend(loc="upper left")
    heat = axes[1].pcolormesh(
        times,
        np.arange(token_scores.shape[1]),
        np.log10(1 + token_scores.T),
        shading="nearest",
        cmap="magma",
        rasterized=True,
    )
    axes[1].set_ylabel("Action token index")
    fig.colorbar(heat, ax=axes[1], label="log10(1 + distance)", pad=0.01)
    colors = {"left": "#147d64", "right": "#b35e13", "eef": "#147d64"}
    for arm in arms:
        values = [np.nan if row[f"acc_{arm}"] is None else row[f"acc_{arm}"] for row in rows]
        raw = [np.nan if row[f"acc_raw_{arm}"] is None else row[f"acc_raw_{arm}"] for row in rows]
        axes[2].plot(times, raw, color=colors[arm], alpha=0.2, linewidth=0.6)
        axes[2].plot(times, values, color=colors[arm], linewidth=1.4, label=f"{arm}: ACC EMA")
        axes[3].plot(
            times,
            [np.nan if row[f"mae_mm_{arm}"] is None else row[f"mae_mm_{arm}"] for row in rows],
            color=colors[arm],
            linewidth=1.0,
            label=f"{arm}: overlap XYZ MAE",
        )
    if acc_threshold is not None:
        axes[2].axhline(acc_threshold, color="#b42318", linestyle="--", label="ACC threshold")
    else:
        axes[2].text(
            0.99,
            0.95,
            "No calibrated ACC threshold",
            transform=axes[2].transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color="#765b19",
        )
    axes[2].set_ylabel("Normalized inconsistency")
    axes[2].legend(loc="upper left")
    axes[3].set_ylabel("Mean absolute error (mm)")
    axes[3].set_xlabel("Recorded episode time (s)")
    axes[3].legend(loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.15)
        if failure_time is not None:
            if not times[0] <= failure_time <= times[-1]:
                raise ValueError("failure_time must lie within the plotted observations")
            ax.axvline(failure_time, color="#b42318", linestyle=":", linewidth=1)
    note = "Faint lines: raw ACC. First ACC query is unavailable. Peaks are not verified failure onset."
    if llmd.n_samples <= llmd.feature_shape[1]:
        note += f"\nReference N={llmd.n_samples} < feature dimension+1; covariance relies on ridge regularization."
    fig.supxlabel(note, fontsize=10)
    fig.savefig(output / "scores.png", dpi=170)
    fig.savefig(output / "scores.svg")
    plt.close(fig)
    with (output / "scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "episode_id": episode["episode_id"],
        "episode_artifact": str(episode_path),
        "reference_artifact": str(stats_path),
        "observations": len(rows),
        "duration_s": float(times[-1] - times[0]),
        "llmd_peak": rows[peak_llmd],
        "acc_peak": rows[peak_acc],
        "llmd_threshold": llmd.threshold,
        "acc_threshold": acc_threshold,
        "ema_alpha": ema_alpha,
        "v_min_m": v_min,
        "feature_metadata": episode["metadata"],
        "reference_metadata": llmd.metadata,
        "reference_samples": llmd.n_samples,
        "feature_shape": list(llmd.feature_shape),
        "covariance_regularization": llmd.regularization,
        "reference_rank_deficient_before_regularization": llmd.n_samples <= llmd.feature_shape[1],
        "query_interval_s": {
            "min": float(np.diff(times).min()),
            "max": float(np.diff(times).max()),
        },
        "provenance": json.loads(str(episode.get("provenance", np.array("{}")).item())),
        "evaluation_notes": str(episode.get("evaluation_notes", np.array("")).item()),
        "failure_time_s": failure_time,
        "arms": arms,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    if bundle_path is not None:
        with np.load(bundle_path, allow_pickle=False) as bundle:
            frame_indices = bundle["frame_indices"]
            selected = sorted({0, peak_llmd, peak_acc, len(rows) - 1})
            fig, axes = plt.subplots(
                3, len(selected), figsize=(4 * len(selected), 8), squeeze=False, layout="constrained"
            )
            for col, query in enumerate(selected):
                matches = np.flatnonzero(frame_indices == rows[query]["frame_index"])
                if len(matches) != 1:
                    raise ValueError("Bundle frames do not match extracted feature frames")
                for row, camera in enumerate(["base", "left_wrist", "right_wrist"]):
                    frame = bundle[f"images_{camera}"][matches[0]]
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 1)
                    axes[row, col].imshow(frame)
                    axes[row, col].axis("off")
                    axes[row, col].set_title(f"{camera} | t={times[query]:.1f}s", fontsize=10)
            fig.suptitle(f"{episode_name} | recorded camera frames at start / score peaks / end", fontsize=15)
            fig.savefig(output / "frames.png", dpi=140)
            plt.close(fig)
    print(
        json.dumps(
            {
                "episode": episode["episode_id"],
                "llmd_peak_time_s": float(times[peak_llmd]),
                "acc_peak_time_s": float(times[peak_acc]),
                "output": str(output),
            }
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", help="Optional observation bundle for matching camera snapshots")
    parser.add_argument("--ema-alpha", type=float, default=0.9)
    parser.add_argument("--v-min", type=float, default=0.001)
    parser.add_argument("--acc-threshold", type=float)
    parser.add_argument(
        "--failure-time", type=float, help="Annotated failure onset, never inferred from episode outcome"
    )
    args = parser.parse_args()
    render(
        args.episode,
        args.stats,
        args.output,
        ema_alpha=args.ema_alpha,
        v_min=args.v_min,
        acc_threshold=args.acc_threshold,
        bundle_path=args.bundle,
        failure_time=args.failure_time,
    )


if __name__ == "__main__":
    main()
