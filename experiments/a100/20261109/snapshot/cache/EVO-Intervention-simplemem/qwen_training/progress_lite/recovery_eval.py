# ruff: noqa: RUF001, RUF002, RUF003
"""插螺栓回退诊断：无回退对照、裁剪开头、非典型时间。用法见 --help。"""

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

CONTROL = "insert-screw"
RECOVERY = "insert-screw-recover"
FPS = 30  # source frame indices and dense labels of these two tasks


def median(values):
    return float(np.median(values)) if len(values) else None


def drop_window(run, center=None):
    """最大约一秒下降：返回（进度百分点，窗口中点秒数）。正数表示下降。"""
    t, p = np.asarray(run["t"]), np.asarray(run["p"]) * 100
    end = np.searchsorted(t, t + 1 - 1e-6)
    begin = np.flatnonzero(end < len(t))
    end = end[begin]
    valid = t[end] - t[begin] <= 1.21  # allow one sampling interval at 5 Hz
    mid = (t[end] + t[begin]) / 2
    if center is not None:
        covered = mid[valid]
        if not len(covered) or covered[0] > center - 1 or covered[-1] < center + 1:
            return None
        valid &= abs(mid - center) <= 1
    if not valid.any():
        return None
    drops = (p[begin] - p[end])[valid]
    i = drops.argmax()
    return float(drops[i]), float(mid[valid][i])


def stats(windows):
    drops = [w[0] for w in windows if w is not None]
    return {"eligible": len(drops), "drops_ge5pp": sum(x >= 5 for x in drops), "median_drop_pp": median(drops)}


def summarize(rows):
    controls = [r["runs"]["0"] for r in rows if r["task"] == CONTROL]
    recovery = [r for r in rows if r["task"] == RECOVERY]
    matched = [run for run in controls if all(drop_window(run, t) is not None for t in (10, 15))]
    control = {
        "all": stats(map(drop_window, controls)),
        **{f"matched_{t}s": stats(drop_window(run, t) for run in matched) for t in (10, 15)},
    }
    crops = {}
    for cut in ("0", "5", "10"):
        shifts, event_errors, clock_errors, events, clocks = [], [], [], [], []
        for row in recovery:
            if cut not in row["runs"]:
                continue
            run = row["runs"][cut]
            offset = run["cut"]
            times = [e - offset for e in row["events"] if e >= offset]
            events.extend(drop_window(run, e) for e in times)
            if all(abs(e - 15) > 2.1 for e in times):
                clocks.append(drop_window(run, 15))
            orig, new = drop_window(row["runs"]["0"]), drop_window(run)
            if cut != "0" and orig and new and min(orig[0], new[0]) >= 5 and orig[1] > offset + 1:
                shift = orig[1] - new[1]
                shifts.append(shift)
                event_errors.append(abs(shift - offset))
                clock_errors.append(abs(shift))
        crops[cut] = {
            "at_event": stats(events),
            "at_15s_without_event": stats(clocks),
            "paired": len(shifts),
            "median_advance_s": median(shifts),
            "median_event_timing_error_s": median(event_errors),
            "median_clock_timing_error_s": median(clock_errors),
            "closer_to_event_shift": sum(a < b for a, b in zip(event_errors, clock_errors, strict=True)),
        }
    atypical = {}
    for name in ("early", "late"):
        events, clocks, changes, targets = [], [], [], []
        videos = set()
        for row in recovery:
            times = [e for e in row["events"] if (e < 12 if name == "early" else e > 24)]
            if not times:
                continue
            videos.add(row["video"])
            run = row["runs"]["0"]
            if all(abs(e - 15) > 2.1 for e in row["events"]):
                clocks.append(drop_window(run, 15))
            for e in times:
                events.append(drop_window(run, e))
                if run["t"][0] <= e - 0.2 and e + 1 <= run["t"][-1]:
                    for key, out in (("p", changes), ("gt", targets)):
                        left, right = np.interp([e - 0.2, e + 1], run["t"], run[key])
                        out.append(100 * (right - left))
        atypical[name] = {
            "events": len(events),
            "source_videos": len(videos),
            "at_event": stats(events),
            "at_15s_without_event": stats(clocks),
            "signed_change_n": len(changes),
            "median_prediction_change_pp": median(changes),
            "median_label_change_pp": median(targets),
        }
    return {
        "control_episodes": len(controls),
        "recovery_episodes": len(recovery),
        "recovery_events": sum(len(r["events"]) for r in recovery),
        "no_rollback": control,
        "crop": crops,
        "atypical": atypical,
    }


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def evaluate(args):
    import fcntl

    import torch

    from .config import Config
    from .data import Trajectories, reorder
    from .model import ProgressLite, autocast

    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{f.name: saved["config"][f.name] for f in fields(Config) if f.name in saved["config"]})
    for name in ("model", "data", "labels", "device"):
        if getattr(args, name):
            setattr(cfg, name, getattr(args, name))
    ds = Trajectories(cfg.labels, cfg.data, "val")
    candidates = [
        (i, task, ep, ds.labels / task / f"episode{ep}.json")
        for i, (task, ep) in enumerate(ds.episodes)
        if task in (CONTROL, RECOVERY)
    ]
    if {task for _, task, _, _ in candidates} != {CONTROL, RECOVERY}:
        raise ValueError("需要同时包含 insert-screw 与 insert-screw-recover 的验证标签；请检查 --labels")
    identity = hashlib.sha256()
    with args.checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            identity.update(block)
    labels_hash = hashlib.sha256()
    for _, task, ep, path in candidates:
        labels_hash.update(f"{task}/{ep}\0".encode() + path.read_bytes())
    manifest = {
        "protocol": 1,
        "checkpoint_sha256": identity.hexdigest(),
        "iteration": saved["iteration"],
        "labels_sha256": labels_hash.hexdigest(),
        "model": cfg.model,
        "data": cfg.data,
        "device": cfg.device,
    }
    index = args.out / "manifest.json"
    if index.exists() and json.loads(index.read_text()) != manifest:
        raise ValueError("输出目录属于其他权重或数据，请使用新的 --out")
    write_json(index, manifest)
    dest = args.out / "rollouts"
    dest.mkdir(exist_ok=True)
    model = ProgressLite.from_pretrained(cfg.model, cfg, cfg.device)
    model.load_trainable(saved["model"])
    del saved
    model.eval()
    rows = []
    with torch.inference_mode():
        for i, task, ep, path in candidates:
            rec = json.loads(path.read_text())
            idx = np.array([f["index"] for f in rec["frames"]])
            dense = np.asarray(rec["dense_progress"])
            offset = rec["progress_start"]
            times = (idx - idx[0]) / FPS
            events = (np.flatnonzero(np.diff(dense) < -0.02) + 1 + offset - idx[0]) / FPS
            if task == CONTROL and (rec["rollback_transitions"] or len(events)):
                continue
            if task == RECOVERY and not rec["rollback_transitions"]:
                continue
            file = dest / f"{task}_{ep}.json"
            if not file.exists():
                sample = ds.sample(i, cfg)
                gt = dense[idx - offset]  # 恢复任务的进度允许下降，不能使用时间比例代理。
                row = dict(task=task, episode=ep, video=rec["source_video_id"], events=events.tolist(), runs={})
                for cut in (0,) if task == CONTROL else (0, 5, 10):
                    j = int(np.searchsorted(times, cut))
                    if j >= len(times) - 5:
                        continue
                    with autocast(cfg):
                        pred = model.expected(model.rollout(reorder(sample, list(range(j, len(times)))))).float().cpu()
                    row["runs"][str(cut)] = dict(
                        t=(times[j:] - times[j]).tolist(), cut=float(times[j]), p=pred.tolist(), gt=gt[j:].tolist()
                    )
                if "0" not in row["runs"]:
                    continue
                write_json(file, row)
            rows.append(json.loads(file.read_text()))
            print(task, ep, flush=True)
    result = summarize(rows)
    write_json(args.out / "summary.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    for name in ("model", "labels", "data", "device"):
        parser.add_argument(f"--{name}", help="覆盖 checkpoint 配置")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
