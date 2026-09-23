"""Descriptive comparison of labeled trajectories; does not calibrate thresholds."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def describe(times, values, mask):
    usable = mask & np.isfinite(values)
    if not usable.any():
        return None
    data, stamps = values[usable], times[usable]
    peak = int(np.argmax(data))
    return {
        "samples": len(data),
        "median": float(np.median(data)),
        "p95": float(np.quantile(data, 0.95)),
        "max": float(data[peak]),
        "peak_time_s": float(stamps[peak]),
    }


def compare(reports, labels, output):
    if len(reports) != len(labels) or len(reports) < 2:
        raise ValueError("Provide at least two reports and one label for each")
    records = []
    for report_path, label in zip(reports, labels, strict=True):
        report = Path(report_path)
        summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
        with (report / "scores.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any(row["episode_id"] != summary["episode_id"] for row in rows):
            raise ValueError("Mismatched CSV and summary episode")
        times = np.array([float(row["time_s"]) for row in rows])
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Invalid score timeline")
        fields = [
            "llmd",
            "acc_left",
            "acc_right",
            "acc_max",
            "acc_raw_left",
            "acc_raw_right",
            "mae_mm_left",
            "mae_mm_right",
        ]
        values = {key: np.array([float(row[key]) if row[key] else np.nan for row in rows]) for key in fields}
        values["mae_max_mm"] = np.fmax(values["mae_mm_left"], values["mae_mm_right"])
        values["acc_raw_max"] = np.fmax(values["acc_raw_left"], values["acc_raw_right"])
        if not np.allclose(values["acc_max"], np.fmax(values["acc_left"], values["acc_right"]), equal_nan=True):
            raise ValueError("ACC aggregation differs between reports")
        if records:
            for key in [
                "feature_metadata",
                "reference_metadata",
                "reference_samples",
                "covariance_regularization",
                "ema_alpha",
                "v_min_m",
                "query_interval_s",
            ]:
                if summary[key] != records[0]["summary"][key]:
                    raise ValueError(f"Reports use different evaluation conditions: {key}")
        records.append({"label": label, "report": str(report), "summary": summary, "times": times, "values": values})
    common_start = max(record["times"][0] for record in records)
    common_end = min(record["times"][-1] for record in records)
    if common_end <= common_start:
        raise ValueError("No common observation window")
    statistics = []
    for record in records:
        times = record["times"]
        statistics.append(
            {
                "label": record["label"],
                "episode_id": record["summary"]["episode_id"],
                "report": record["report"],
                "sampled_duration_s": float(times[-1] - times[0]),
                "provenance": record["summary"]["provenance"],
                "full": {
                    key: describe(times, values, np.ones(len(times), dtype=bool))
                    for key, values in record["values"].items()
                },
                "common_window": {
                    key: describe(times, values, (times >= common_start) & (times <= common_end))
                    for key, values in record["values"].items()
                },
            }
        )
    maxima = {
        "llmd": max(np.nanmax(record["values"]["llmd"]) for record in records),
        "acc": max(np.nanmax(record["values"]["acc_max"]) for record in records),
        "mae_mm": max(np.nanmax(record["values"]["mae_max_mm"]) for record in records),
    }
    limits = {
        key: float(np.ceil(value * 1.05 / step) * step)
        for (key, value), step in zip(maxima.items(), [10000, 0.5, 10], strict=True)
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "common_window_s": [float(common_start), float(common_end)],
        "axis_limits": limits,
        "records": statistics,
        "interpretation": "Descriptive case comparison only. No calibrated threshold, accuracy, or failure-onset claim.",
    }
    (output / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "axis_limits.json").write_text(json.dumps(limits, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(
        len(records), 3, figsize=(15, 3 * len(records)), sharey="col", squeeze=False, layout="constrained"
    )
    for row, record in enumerate(records):
        times, values = record["times"], record["values"]
        for col, fields in enumerate([["llmd"], ["acc_left", "acc_right"], ["mae_mm_left", "mae_mm_right"]]):
            ax = axes[row, col]
            colors = ["#2359a4"] if col == 0 else ["#147d64", "#b35e13"]
            for field, color in zip(fields, colors, strict=True):
                ax.plot(times, values[field], linewidth=1.2, color=color, label=field)
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(0, limits[["llmd", "acc", "mae_mm"][col]])
            ax.set_xlabel("Recorded time (s)")
            ax.grid(alpha=0.15)
            ax.set_title(
                record["label"] if col == 0 else ["", "ACC EMA: left / right", "XYZ MAE (mm): left / right"][col],
                fontsize=11,
            )
            if col == 0:
                ax.set_ylabel("LLMD squared distance")
    fig.suptitle(
        "Two successful demonstrations vs two autonomous failures\nIdentical model/reference; same vertical scales. Horizontal axes follow each episode duration.",
        fontsize=15,
    )
    fig.savefig(output / "comparison.png", dpi=170)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    print(json.dumps({"common_window_s": result["common_window_s"], "axis_limits": limits, "episodes": len(records)}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    compare(args.reports, args.labels, args.output)


if __name__ == "__main__":
    main()
