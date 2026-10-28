# ruff: noqa: RUF001, RUF002, RUF003
"""把一条 episode 的逐步输出导出成 JSON，用 viewer.html 打开查看每一步的 [STG]、[CNT]、p。

python -m qwen_training.progress.visualize --checkpoint outputs/progress/run/training/best.pt \
    --task swap_T --episode 49 [--split val] [--keyframes model] [--out swap_T_49.json]

JSON 结构（viewer.html 读取）：
- task、episode、instruction、steps（各项文字）、cameras、checkpoint、iteration、keyframes、max_count、count_mode；
- frames：每一步一项，含 step（第几步）、index（原始帧号）、time（秒）、keyframe（这一步之后是否标为关键帧）、
  images（各视角的 JPEG data URL）、
  prediction：stage（预测项，J 表示不在列表里）、stage_probs [J+1]、count [J]（见 evaluate.predicted_counts）、
  count_probs [J][R]、expected [J]（Σ_r P(n_j ≥ r)）、progress；count_mode=event 时 count_probs 为预测次数的
  "至少 r 次"编码、expected 即预测次数；event 的直接输出是 change_probs [J][3]（−1、0、+1 的概率）与
  change [J]，fed [J] 为这一步输入的计数，count 为 clamp(fed + change)；
  truth：stage、count [J]、progress；event 另有 change（标注相邻帧之差）、
  correction（训练目标：clamp(真值 − fed, −1, 1)）。
"""

import argparse
import base64
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_training.datasets.rmbench.labels import MAX_COUNT
from qwen_training.progress.evaluate import predicted_counts
from qwen_training.progress.loss import autocast
from qwen_training.progress.model import cumulative


def jpeg(image, quality=80):
    import cv2

    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG 编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


@torch.no_grad()
def export(model, sample, keyframes="model", indices=None, fps=None, cameras=None):
    """逐步推理整条 sample，返回可写成 JSON 的字典。indices、fps 给出原始帧号和帧率，用于显示时间。"""
    cfg = model.cfg
    was_training = model.training
    model.eval()
    with autocast(cfg):
        outputs = model.rollout(sample, keyframes=keyframes)
    model.train(was_training)
    stage = torch.stack([out["stage"] for out in outputs]).float().softmax(-1)
    counts = predicted_counts(model, outputs)                                          # [T, J]
    progress = (F.softmax(torch.stack([out["progress"] for out in outputs]).float(), -1) * model.bins).sum(-1)
    # 与 rollout 相同的规则：标注模式用样本给的关键帧；模型模式为回传给 [CNT] 的状态在这一步有变化
    # （level：任一概率变化超过 δ；event：预测次数与这一步回传的次数不同）。
    if model.event:
        change = torch.stack([out["count"] for out in outputs]).float().softmax(-1)  # [T, J, 3]
        fed = torch.stack([out["fed"] for out in outputs])
        truth = sample["count"].sum(-1).long().to(fed.device)
        truth_change = torch.cat([truth[:1], truth[1:] - truth[:-1]])
        correction = (truth - fed).clamp(-1, 1)
        count = cumulative(counts)
        changed = (counts != fed).any(-1)
        changed[0] = False                                                              # rollout 从第 1 步起才判断
    else:
        count = torch.stack([out["count"] for out in outputs]).float().sigmoid()       # [T, J, R]
        changed = torch.cat([torch.zeros(1, dtype=torch.bool, device=count.device),
                             (count[1:] - count[:-1]).flatten(1).abs().amax(-1) > cfg.delta])
    marked = {"label": sample["keyframes"].tolist(), "model": changed.tolist()}.get(keyframes, [False] * len(outputs))
    indices = indices or list(range(len(outputs)))
    frames = []
    for t, views in enumerate(sample["frames"]):
        frames.append({
            "step": t, "index": indices[t], "time": round(indices[t] / fps, 3) if fps else t,
            "keyframe": bool(marked[t]),
            "images": [jpeg(view) for view in views],
            "prediction": {"stage": int(stage[t].argmax()), "stage_probs": stage[t].round(decimals=4).tolist(),
                           "count": counts[t].tolist(),
                           "count_probs": count[t].round(decimals=4).tolist(),
                           "expected": count[t].sum(-1).round(decimals=3).tolist(),
                           **({"change_probs": change[t].round(decimals=4).tolist(),
                               "change": (change[t].argmax(-1) - 1).tolist(), "fed": fed[t].tolist()}
                              if model.event else {}),
                           "progress": round(progress[t].item(), 4)},
            "truth": {"stage": int(sample["stage"][t]), "count": sample["count"][t].sum(-1).int().tolist(),
                      "progress": round(float(sample["progress"][t]), 4),
                      **({"change": truth_change[t].tolist(), "correction": correction[t].tolist()}
                         if model.event else {})}})
    return {"task": sample["task"], "episode": sample.get("episode"), "instruction": sample["instruction"],
            "steps": sample["steps"], "cameras": cameras or [f"view{i}" for i in range(len(sample["frames"][0]))],
            "keyframes": keyframes, "max_count": MAX_COUNT, "count_mode": "event" if model.event else "level",
            "frames": frames}


def main(argv=None):
    from qwen_training.datasets.rmbench.labels import CAMERAS
    from qwen_training.progress.config import Config
    from qwen_training.progress.data import RMBench
    from qwen_training.progress.model import ProgressModel

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--keyframes", default="model", choices=("model", "label", "none"),
                        help="model 为实际推理；label 用标注关键帧（oracle）")
    parser.add_argument("--out", help="默认写在 checkpoint 旁边：<task>_ep<episode>_<keyframes>.json")
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**checkpoint["config"])
    model = ProgressModel.from_pretrained(cfg.model, cfg, cfg.device)
    model.load_trainable(checkpoint["model"])
    dataset = RMBench(cfg.labels, cfg.data, args.split, [args.task])
    if (args.task, args.episode) not in dataset.episodes:
        raise ValueError(f"{args.split} 划分里没有 {args.task} episode {args.episode}")
    sample = dataset.sample(dataset.episodes.index((args.task, args.episode)), cfg)
    record = json.loads((dataset.labels / args.task / f"episode{args.episode}.json").read_text())
    result = export(model, sample, args.keyframes, [frame["index"] for frame in record["frames"]],
                    record["source_fps"], list(CAMERAS))
    result.update(checkpoint=str(args.checkpoint), iteration=checkpoint["iteration"])
    out = Path(args.out or Path(args.checkpoint).with_name(f"{args.task}_ep{args.episode}_{args.keyframes}.json"))
    out.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"wrote {out}（{len(result['frames'])} 步），用 qwen_training/progress/viewer.html 打开")


if __name__ == "__main__":
    main()
