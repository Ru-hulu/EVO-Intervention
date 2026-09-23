"""ACC-only scoring and deterministic descriptive ranking of recorded episodes."""

import numpy as np

from .acc import ACCDetector


def score_episode(episode_id, left, right, frames, times, *, alpha=0.9, v_min=0.01):
    left, right = np.asarray(left), np.asarray(right)
    frames, times = np.asarray(frames), np.asarray(times)
    count = len(frames)
    if count < 2 or left.ndim != 3 or left.shape != right.shape or left.shape[0] != count or left.shape[2] != 3:
        raise ValueError("Need at least two aligned [T,H,3] position chunks")
    if frames.shape != (count,) or times.shape != (count,) or not np.isfinite(times).all():
        raise ValueError("Invalid frame/time arrays")
    if frames.dtype.kind not in "iu" or np.any(np.diff(frames) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Frame indices and timestamps must increase")
    detectors = {arm: ACCDetector(alpha, v_min) for arm in ("left", "right")}
    rows = []
    for index in range(count):
        row = {
            "episode_id": episode_id,
            "query_index": index,
            "frame_index": int(frames[index]),
            "time_s": float(times[index]),
        }
        for arm, positions in (("left", left), ("right", right)):
            result = detectors[arm].update(
                positions[index], None if index == 0 else int(frames[index] - frames[index - 1])
            )
            row[f"acc_{arm}"] = result.score
            row[f"acc_raw_{arm}"] = result.raw_score
            row[f"mae_mm_{arm}"] = None if not result.ready else float(np.mean(result.per_axis_mae) * 1000)
        row["acc_max"] = None if index == 0 else max(row["acc_left"], row["acc_right"])
        rows.append(row)
    return rows


def describe_rows(rows):
    values = np.array([r["acc_max"] for r in rows if r["acc_max"] is not None], dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("No finite ACC observations for ranking")
    return {
        "valid_samples": len(values),
        "acc_p95": float(np.quantile(values, 0.95)),
        "acc_median": float(np.median(values)),
        "acc_max": float(values.max()),
    }


def rank_records(records):
    if len(records) < 4 or len({r["episode_id"] for r in records}) != len(records):
        raise ValueError("Ranking requires at least four unique episodes")
    if any(not np.isfinite([r["acc_p95"], r["acc_median"]]).all() for r in records):
        raise ValueError("Ranking scores must be finite")
    ordered = sorted(records, key=lambda r: (r["acc_p95"], r["acc_median"], r["episode_id"]))
    return ordered, {"lowest": ordered[:2], "highest": list(reversed(ordered[-2:]))}
