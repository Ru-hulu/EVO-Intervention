# ruff: noqa: RUF001, RUF002, RUF003
"""时钟诊断（proposal_progress_v10）：p 跟画面内容走，还是跟输入步数走。

每条验证轨迹跑基线（原序、停顿、0.5 倍速、2 倍速）和 S 重置（原序、停顿），
逐帧结果存到 --out/rollouts，中断后重跑会跳过已完成的部分；跑完写 --out/summary.json。

python -m qwen_training.progress_lite.clock_test --checkpoint <权重> --out <目录> [--model ... --labels ... --data ...]
"""

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from qwen_training.progress_lite.config import Config
from qwen_training.progress_lite.data import Trajectories, reorder
from qwen_training.progress_lite.model import ProgressLite, autocast

PAUSE = 20  # 停顿测试每处重复的步数，5 Hz 下 4 秒
SPEEDS = {"speed0.5": 0.5, "speed2": 2.0}
PLAN = {"baseline": ("normal", "pause", *SPEEDS), "reset": ("normal", "pause")}
MIN_OFFSET = 5  # 步数和画面至少错开这么多帧，两种预测才有区分力


def pause_order(n):
    """在 1/4、1/2、3/4 处各把一帧重复 PAUSE 步。"""
    order = []
    for t in range(n):
        order += [t] * (PAUSE + 1 if t in (n // 4, n // 2, 3 * n // 4) else 1)
    return order


@torch.no_grad()
def evaluate(model, dataset, cfg, episodes=0):
    """训练中的验证：均匀抽 episodes 条（0 为全部），报告原序、0.5 倍速、2 倍速的 P MAE，mae_speed 是后两者的均值。"""
    was_training = model.training
    model.eval()
    count = min(episodes, len(dataset)) if episodes else len(dataset)
    errors = defaultdict(list)
    for index in (i * len(dataset) // count for i in range(count)):
        for name, speed in {"normal": None, **SPEEDS}.items():
            sample = dataset.sample(index, cfg, speed=speed)
            with autocast(cfg):
                p = model.expected(model.rollout(sample))
            errors[name].append((p.cpu() - sample["progress"]).abs().mean().item())
    model.train(was_training)
    metrics = {f"val/mae_{name}": float(np.mean(values)) for name, values in errors.items()}
    return {**metrics, "val/mae_speed": float(np.mean([metrics[f"val/mae_{name}"] for name in SPEEDS]))}


def reset_steps(model):
    """替换 steps：每个 chunk 开头把 S 恢复成前缀写完时的值。"""
    original = ProgressLite.steps.__get__(model)

    def steps(state, frames, start):
        if start == 0:
            state.prefix_s = {k: v.clone() for k, v in state.recurrent.items()}
        else:
            state.recurrent = {k: v.clone() for k, v in state.prefix_s.items()}
        return original(state, frames, start)

    return steps


def rollouts(model, dataset, cfg, out, episodes):
    """跑完 PLAN 里的全部组合。"""
    for index in range(episodes or len(dataset)):
        sample = dataset.sample(index, cfg)
        sequences = {"normal": sample, "pause": reorder(sample, pause_order(len(sample["frames"])))}
        for name, speed in SPEEDS.items():
            sequences[name] = dataset.sample(index, cfg, speed=speed)
        for variant, names in PLAN.items():
            model.steps = reset_steps(model) if variant == "reset" else ProgressLite.steps.__get__(model)
            for name in names:
                path = out / f"{variant}_{name}_{index:04d}.pt"
                if path.exists():
                    continue
                changed = sequences[name]
                with autocast(cfg):
                    p = model.expected(model.rollout(changed))
                torch.save(
                    {
                        "task": sample["task"],
                        "position": changed["positions"].tolist(),
                        "p": p.cpu(),
                        "p_true": changed["progress"],
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
    """各段停顿（同一帧重复超过 PAUSE 步）里 p 的平均变化。"""
    position, drifts, start = run["position"], [], 0
    for i in range(1, len(position) + 1):
        if i == len(position) or position[i] != position[start]:
            if i - start > PAUSE:
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
        row |= {f"drift/{key}": pause_drift(run) for key, run in group.items() if key.endswith("/pause")}
        row |= {f"weight/{name}": content_weight(group[f"baseline/{name}"], normal) for name in SPEEDS}
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
    paths = {key: value for key in ("model", "labels", "data") if (value := getattr(args, key))}
    cfg = Config(**{**saved["config"], **paths})
    cfg = replace(cfg, eval_chunk_size=cfg.window)  # S 重置发生在 chunk 开头，周期等于窗口长度
    out = Path(args.out)
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    dataset = Trajectories(cfg.labels, cfg.data, "val", cfg.suites.split(",") if cfg.suites else None)
    model = ProgressLite.from_pretrained(cfg.model, cfg, cfg.device)
    model.load_trainable(saved["model"])
    model.eval()
    with torch.inference_mode():
        rollouts(model, dataset, cfg, out / "rollouts", args.episodes)
    result = summarize(out / "rollouts")
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
