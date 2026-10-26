# ruff: noqa: RUF001, RUF002
"""验证：整条 episode 逐步推理（与实际推理相同），用评估示范，报告总体和逐任务的指标。

python -m qwen_training.progress.evaluate --checkpoint outputs/progress/run/training/best.pt [--split val]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_training.progress.loss import autocast, sample_loss

METRICS = ("loss", "stage_loss", "count_loss", "progress_loss", "stage_acc", "count_acc", "progress_mae")


@torch.no_grad()
def evaluate(model, dataset, cfg, episodes=0):
    """episodes 为 0 时验证全部，否则均匀抽取这么多条。

    - 关键帧分两种：oracle 用无扰动的标注状态变化帧，model 用模型自己按 δ 触发的（即实际推理）。
    - loss 与训练同一公式，按帧加权平均；stage_acc 不扣除边界帧。
    - count_acc：每帧每项预测的次数与真值完全相同的比例。预测次数为 P(n_j ≥ r) > 0.5 的个数，
      再沿时间取累计最大：目前只有成功轨迹，完成次数只增不减（proposal 3.5）。
    返回 {"overall": {"val/<模式>/<指标>": 值}, "per_task": {任务: {"<模式>/<指标>": 值}}}。
    """
    was_training = model.training
    model.eval()
    count = len(dataset) if episodes <= 0 else min(episodes, len(dataset))
    indices = [i * len(dataset) // count for i in range(count)]
    modes = {"oracle": "label", "model": "model"} if cfg.keyframes != "none" else {"model": "none"}
    sums = defaultdict(lambda: defaultdict(float))       # (模式, 任务) → 累计量
    for name, mode in modes.items():
        for index in indices:
            sample = dataset.sample(index, cfg)
            with autocast(cfg):
                outputs = model.rollout(sample, keyframes=mode)
                loss, parts = sample_loss(model, outputs, sample, cfg)
            logits = {key: torch.stack([out[key] for out in outputs]).float() for key in ("stage", "count", "progress")}
            labels = {key: sample[key].to(model.special.device) for key in ("stage", "count", "progress")}
            frames = len(outputs)
            p = (F.softmax(logits["progress"], -1) * model.bins).sum(-1)
            counts = (logits["count"] > 0).sum(-1).cummax(0).values
            metrics = {"frames": frames, "items": labels["count"][..., 0].numel(), "loss": loss.item() * frames,
                       **{f"{part}_loss": parts[part] * frames for part in ("stage", "count", "progress")},
                       "stage_acc": (logits["stage"].argmax(-1) == labels["stage"]).sum().item(),
                       "count_acc": (counts == labels["count"].sum(-1)).sum().item(),
                       "progress_mae": (p - labels["progress"]).abs().sum().item()}
            for key in ((name, sample["task"]), (name, None)):
                for metric, value in metrics.items():
                    sums[key][metric] += value
    model.train(was_training)

    def average(total):
        return {metric: total[metric] / total["items" if metric == "count_acc" else "frames"] for metric in METRICS}

    overall, per_task = {}, defaultdict(dict)
    for (name, task), total in sums.items():
        for metric, value in average(total).items():
            if task is None:
                overall[f"val/{name}/{metric}"] = value
            else:
                per_task[task][f"{name}/{metric}"] = value
    return {"overall": overall, "per_task": dict(sorted(per_task.items())), "episodes": count}


def main(argv=None):
    from qwen_training.progress.config import Config
    from qwen_training.progress.data import RMBench
    from qwen_training.progress.model import ProgressModel

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--episodes", type=int, default=0, help="0 为该划分的全部 episode")
    parser.add_argument("--out", help="结果 JSON，默认写在 checkpoint 旁边")
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**checkpoint["config"])
    model = ProgressModel.from_pretrained(cfg.model, cfg, cfg.device)
    model.load_trainable(checkpoint["model"])
    tasks = cfg.tasks.split(",") if cfg.tasks else None
    result = evaluate(model, RMBench(cfg.labels, cfg.data, args.split, tasks), cfg, args.episodes)
    result.update(checkpoint=str(args.checkpoint), iteration=checkpoint["iteration"], split=args.split)
    out = Path(args.out or Path(args.checkpoint).with_name(f"{Path(args.checkpoint).stem}_{args.split}_eval.json"))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["overall"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
