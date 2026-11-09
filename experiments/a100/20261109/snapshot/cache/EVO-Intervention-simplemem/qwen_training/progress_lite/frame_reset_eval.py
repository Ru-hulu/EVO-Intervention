# ruff: noqa: RUF002
"""对参考回退轨迹逐帧重置状态，保留任务指令与当前三视角画面。"""

import argparse
import fcntl
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from .recovery_eval import drop_window, median, write_json


def evaluate(args):
    import torch

    from .config import Config
    from .data import Trajectories
    from .model import ProgressLite, autocast

    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.cuda.set_per_process_memory_fraction(
        args.max_memory_gib * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**{f.name: saved["config"][f.name] for f in fields(Config) if f.name in saved["config"]})
    cfg.model, cfg.labels, cfg.data = args.model, args.labels, args.data
    reference = json.loads((args.reference / "manifest.json").read_text())
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert digest == reference["checkpoint_sha256"], "参考轨迹与权重不匹配"
    baseline = [
        json.loads(p.read_text()) for p in sorted((args.reference / "rollouts").glob("insert-screw-recover_*.json"))
    ]
    assert len(baseline) == 20 and sum(len(r["events"]) for r in baseline) == 28
    manifest = dict(
        protocol="frame_reset_v1",
        checkpoint_sha256=digest,
        iteration=saved["iteration"],
        reference=reference,
        prefix="task instruction",
        frame_index=0,
        model=cfg.model,
        labels=cfg.labels,
        data=cfg.data,
        max_memory_gib=args.max_memory_gib,
        event_window_seconds=2,
        sampling_hz=5,
    )
    index = args.out / "manifest.json"
    if index.exists() and json.loads(index.read_text()) != manifest:
        raise ValueError("输出目录属于另一组评估")
    write_json(index, manifest)
    ds = Trajectories(cfg.labels, cfg.data, "val")
    lookup = {pair: i for i, pair in enumerate(ds.episodes)}
    model = ProgressLite.from_pretrained(cfg.model, cfg, "cpu")
    model.load_trainable(saved["model"])
    model.to(cfg.device)
    del saved
    model.eval()
    dest = args.out / "rollouts"
    dest.mkdir(exist_ok=True)

    def predict(views, instruction):
        with autocast(cfg):
            state = model.start(instruction)
            value = model.expected(model.steps(state, [views], 0)).float().cpu().item()
        del state
        return value

    rows, checked = [], False
    with torch.inference_mode():
        for original in baseline:
            path = dest / f"{original['task']}_{original['episode']}.json"
            if not path.exists():
                sample = ds.sample(lookup[(original["task"], original["episode"])], cfg)
                run = original["runs"]["0"]
                assert len(sample["frames"]) == len(run["t"])
                assert np.allclose(sample["progress"].numpy(), run["gt"])
                if not checked:
                    first = predict(sample["frames"][0], sample["instruction"])
                    predict(sample["frames"][-1], sample["instruction"])
                    repeat = predict(sample["frames"][0], sample["instruction"])
                    assert abs(first - repeat) < 1e-4, "相同帧重置推理不一致"
                    write_json(
                        args.out / "reset_check.json",
                        dict(first=first, after_other_frame=repeat, difference=abs(first - repeat)),
                    )
                    checked = True
                times = np.asarray(run["t"])
                selected = np.flatnonzero(np.any(abs(times[:, None] - np.asarray(original["events"])) <= 2.01, axis=1))
                predictions = []
                for i, index in enumerate(selected):
                    predictions.append(predict(sample["frames"][index], sample["instruction"]))
                    if (i + 1) % 20 == 0:
                        print(original["episode"], "event-window frames", i + 1, "/", len(selected), flush=True)
                reset = dict(
                    t=times[selected].tolist(), p=predictions, gt=np.asarray(run["gt"])[selected].tolist(), cut=0
                )
                write_json(path, {**original, "runs": {"stateful": run, "reset": reset}})
            rows.append(json.loads(path.read_text()))
            print("COMPLETE", original["episode"], len(rows), "/", len(baseline), flush=True)
    events = []
    for row in rows:
        for event in row["events"]:
            result = dict(
                episode=row["episode"],
                video=row["video"],
                time=event,
                category="early" if event < 12 else "late" if event > 24 else "typical",
            )
            for name, run in row["runs"].items():
                window = drop_window(run, event)
                left, right = np.interp([event - 0.2, event + 1], run["t"], run["p"])
                result[name] = dict(drop_pp=window[0] if window else None, change_pp=100 * (right - left))
            left, right = np.interp([event - 0.2, event + 1], row["runs"]["reset"]["t"], row["runs"]["reset"]["gt"])
            result["label_change_pp"] = 100 * (right - left)
            events.append(result)
    summary = {}
    for category in ("all", "early", "typical", "late"):
        group = [e for e in events if category == "all" or e["category"] == category]
        summary[category] = {
            name: dict(
                n=len(group),
                drops_ge5pp=sum(e[name]["drop_pp"] is not None and e[name]["drop_pp"] >= 5 for e in group),
                median_change_pp=median([e[name]["change_pp"] for e in group]),
            )
            for name in ("stateful", "reset")
        }
    write_json(args.out / "summary.json", dict(summary=summary, events=events))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "reference", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("model", "labels", "data"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-memory-gib", type=float, default=11)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
