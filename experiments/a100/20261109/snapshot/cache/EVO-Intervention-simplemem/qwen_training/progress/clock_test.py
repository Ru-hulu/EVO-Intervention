# ruff: noqa: RUF001, RUF002, RUF003
"""时钟诊断（proposal_progress_v9 第 1、5 节）：p 跟画面内容走，还是跟输入步数走。

每条验证轨迹跑基线（原序、停顿、不规则间隔、0.5 倍速、2 倍速）、S 重置和位置平移（原序、停顿），
逐帧结果存到 --out/rollouts，中断后重跑会跳过已完成的部分；跑完写 --out/summary.json。

python -m qwen_training.progress.clock_test --checkpoint <权重> --out <目录> [--model ... --labels ... --data ...]
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qwen_training.progress.config import checkpoint_config
from qwen_training.progress.data import RMBench, speed_positions, warp
from qwen_training.progress.evaluate import pause_order, take
from qwen_training.progress.loss import autocast
from qwen_training.progress.model import ProgressModel

SHIFT_FRAMES = 100  # 位置平移量，5 Hz 下约 20 秒
SPEEDS = {"speed0.5": 0.5, "speed2": 2.0}
PLAN = {
    "baseline": ("normal", "pause", "irregular", *SPEEDS),
    "reset": ("normal", "pause"),
    "shift": ("normal", "pause"),
}
MIN_OFFSET = 5  # 步数和画面至少错开这么多帧，两种预测才有区分力


def ablate(model, variant):
    """替换 teacher_steps：reset 在每个 chunk 开头把 S 恢复成前缀写完时的值，shift 在首帧之后整体后移位置编号。"""
    original = ProgressModel.teacher_steps.__get__(model)

    def steps(state, sample, start, end, keyframes):
        if variant == "reset":
            if start == 0:
                state.prefix_s = {k: v.clone() for k, v in state.recurrent.items()}
            else:
                state.recurrent = {k: v.clone() for k, v in state.prefix_s.items()}
        if variant == "shift" and start == 0:
            before = state.next_position
            head = original(state, sample, 0, 1, keyframes)
            state.next_position += SHIFT_FRAMES * (state.next_position - before)
            return head + (original(state, sample, 1, end, keyframes) if end > 1 else [])
        return original(state, sample, start, end, keyframes)

    return steps


def rollouts(model, dataset, cfg, out, episodes):
    """跑完 PLAN 里的全部组合。position 是每一步在原序里的位置，统计时用来对齐。"""
    warp_cfg = replace(cfg, speed_prob=1.0, pause_prob=1.0, rewind_prob=0.0)
    for index in range(episodes or len(dataset)):
        sample = dataset.sample(index, cfg)
        n = len(sample["frames"])
        orders = {
            "normal": list(range(n)),
            "pause": pause_order(sample)[0],
            "irregular": warp(n, warp_cfg, random.Random(cfg.seed * 100000 + sample["episode"] * 1000)),
        }
        sequences = {name: (take(sample, order), order) for name, order in orders.items()}
        for name, speed in SPEEDS.items():
            sequences[name] = (dataset.sample(index, cfg, speed=speed), speed_positions(n, speed))
        for variant, names in PLAN.items():
            model.teacher_steps = ablate(model, variant)
            for name in names:
                path = out / f"{variant}_{name}_{index:04d}.pt"
                if path.exists():
                    continue
                changed, position = sequences[name]
                with autocast(cfg):
                    outputs = model.rollout(changed, keyframes="none")
                p = (F.softmax(torch.stack([o["progress"] for o in outputs]).float(), -1) * model.bins).sum(-1)
                stage = torch.stack([o["stage"] for o in outputs]).float().argmax(-1)
                torch.save(
                    {
                        "task": sample["task"],
                        "position": position,
                        "p": p.cpu(),
                        "p_true": changed["progress"],
                        "stage": stage.cpu(),
                        "stage_true": changed["stage"],
                    },
                    path.with_suffix(".tmp"),
                )
                path.with_suffix(".tmp").replace(path)
        print(f"{index + 1} {sample['task']}", flush=True)


def content_weight(run, normal):
    """p = w·内容预测 + (1−w)·时钟预测 的最小二乘 w：内容预测取原序同一画面的 p，时钟预测取原序同一步数的 p。"""
    base, position = normal["p"].numpy(), np.array(run["position"], float)
    steps = np.arange(min(len(position), len(base)))
    moved = steps[np.abs(steps - position[steps]) >= MIN_OFFSET]
    if len(moved) < MIN_OFFSET:
        return None
    clock, content = base[moved], base[np.rint(position[moved]).astype(int)]
    return float(
        ((run["p"].numpy()[moved] - clock) * (content - clock)).sum() / max(((content - clock) ** 2).sum(), 1e-9)
    )


def pause_drift(run):
    """各段停顿（同一帧重复超过 20 步）里 p 的平均变化。"""
    position, drifts, start = run["position"], [], 0
    for i in range(1, len(position) + 1):
        if i == len(position) or position[i] != position[start]:
            if i - start > 20:
                drifts.append((run["p"][i - 1] - run["p"][start]).item())
            start = i
    return float(np.mean(drifts)) if drifts else None


def summarize(out):
    """按 suite 汇总，所有比较都在同一批轨迹上配对。"""
    runs = defaultdict(dict)
    for path in sorted(out.glob("*.pt")):
        variant, name, index = path.stem.split("_")
        runs[int(index)][f"{variant}/{name}"] = torch.load(path, weights_only=False)
    rows = defaultdict(lambda: defaultdict(list))
    for group in runs.values():
        if len(group) < sum(map(len, PLAN.values())):
            continue
        normal = group["baseline/normal"]
        row = {f"mae/{key}": (run["p"] - run["p_true"]).abs().mean().item() for key, run in group.items()}
        row |= {f"stage/{key}": (run["stage"] == run["stage_true"]).float().mean().item() for key, run in group.items()}
        row |= {f"drift/{key}": pause_drift(run) for key, run in group.items() if key.endswith("/pause")}
        row["dp/shift"] = (group["shift/normal"]["p"] - normal["p"]).mean().item()
        row |= {f"weight/{name}": content_weight(group[f"baseline/{name}"], normal) for name in ("irregular", *SPEEDS)}
        for suite in (normal["task"].split("/")[0], "all"):
            rows[suite]["episodes"].append(1)
            for key, value in row.items():
                if value is not None:
                    rows[suite][key].append(value)
    return {
        suite: {key: (len(values) if key == "episodes" else float(np.mean(values))) for key, values in row.items()}
        for suite, row in rows.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", help="覆盖 checkpoint 里的路径，换机器时用；--labels、--data 同")
    parser.add_argument("--labels")
    parser.add_argument("--data")
    parser.add_argument("--episodes", type=int, default=0, help="0 为验证集全部")
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint_config(saved["config"], model=args.model, labels=args.labels, data=args.data)
    cfg = replace(cfg, warmup_chunk_size=cfg.window)  # S 重置发生在 chunk 开头，周期等于窗口长度
    out = Path(args.out)
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    dataset = RMBench(
        cfg.labels,
        cfg.data,
        "val",
        cfg.tasks.split(",") if cfg.tasks else None,
        cfg.suites.split(",") if cfg.suites else None,
    )
    model = ProgressModel.from_pretrained(cfg.model, cfg, cfg.device)
    model.load_trainable(saved["model"])
    model.eval()
    with torch.inference_mode():
        rollouts(model, dataset, cfg, out / "rollouts", args.episodes)
    result = summarize(out / "rollouts")
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
