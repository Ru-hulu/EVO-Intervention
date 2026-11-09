# ruff: noqa: RUF001, RUF002, RUF003
"""验证：整条 episode 逐步推理（与实际推理相同），用评估示范，报告总体和逐任务的指标。

python -m qwen_training.progress.evaluate --checkpoint outputs/progress/run/training/best.pt [--split val]
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F

from qwen_training.datasets.rmbench.labels import MAX_COUNT, SAMPLE_FPS
from qwen_training.progress.loss import autocast, sample_loss

BUCKETS = 5   # 按真实次数分桶：0、1、2、3、≥4
METRICS = ("loss", "stage_loss", "count_loss", "progress_loss", "stage_acc", "count_acc", "progress_mae",
           "count_final_acc", "count_acc_n2plus", "count_sync", "count_zero_acc",
           *(f"count_acc_n{b}" for b in range(BUCKETS)))
# 各指标的分母：默认按帧
DENOMINATOR = {"count_acc_n2plus": "items_n2plus", "count_acc": "items", "count_zero_acc": "items",
               "count_final_acc": "episodes", "count_sync": "synced",
               **{f"count_acc_n{b}": f"items_n{b}" for b in range(BUCKETS)}}
PAUSE = 20    # 停顿测试每处重复的步数（5 Hz 下 4 秒）


def sync(scores):
    """各项计数分数 [T, J] 随时间两两的相关系数的平均；接近 1 说明各项共用一条随时间上升的曲线。

    少于两项有变化时为 None。
    """
    scores = scores[:, scores.std(0) > 1e-6]
    if scores.shape[1] < 2:
        return None
    corr = torch.corrcoef(scores.T)
    return corr[~torch.eye(len(corr), dtype=torch.bool, device=corr.device)].mean().item()


def predicted_counts(model, outputs):
    """每帧每项预测的次数 [T, J]：level 为 P(n_j ≥ r) > 0.5 的个数，event 为回传的次数加上预测的变化。"""
    count = torch.stack([out["count"] for out in outputs]).float()
    if model.event:
        fed = torch.stack([out["fed"] for out in outputs])
        return (fed + count.argmax(-1) - 1).clamp(0, MAX_COUNT)
    return (count > 0).sum(-1)


def inference_keyframes(cfg):
    """无关键帧训练沿用 none；启用关键帧的历史实验用模型触发，label 仅作显式 oracle 对照。"""
    return "none" if cfg.keyframes == "none" else "model"


@torch.no_grad()
def evaluate(model, dataset, cfg, episodes=0, keyframes=None):
    """episodes 为 0 时验证全部，否则均匀抽取这么多条。

    - 关键帧分两种：oracle 用无扰动的标注状态变化帧，model 用模型自己按 δ 触发的（即实际推理）。
    - keyframes 省略时比较 oracle/model；指定时只运行该模式。
    - loss 与训练同一公式，按帧加权平均；stage_acc 不扣除边界帧。
    - count_acc：每帧每项预测的次数（见 predicted_counts）与真值完全相同的比例；count_acc_n<b> 按真实次数分桶
      （n4 为 ≥ 4）；count_acc_n2plus 按所有真实次数 ≥ 2 的帧/项目合并；
      count_final_acc：末帧各项全部数对的 episode 比例；count_sync：见 sync()；
      count_zero_acc：一直预测 0 次能得到的 count_acc（平凡基线）。
    返回 {"overall": {"val/<模式>/<指标>": 值}, "per_task": {任务: {"<模式>/<指标>": 值}}}。
    """
    was_training = model.training
    model.eval()
    count = len(dataset) if episodes <= 0 else min(episodes, len(dataset))
    indices = [i * len(dataset) // count for i in range(count)]
    modes = {"oracle": "label", "model": "model"} if cfg.keyframes != "none" else {"model": "none"}
    if keyframes is not None:
        modes = {"oracle" if keyframes == "label" else "model": keyframes}
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
            counts, truth = predicted_counts(model, outputs), labels["count"].sum(-1)
            hits, buckets = counts == truth, truth.clamp(max=BUCKETS - 1)
            score = sync(counts.float() if model.event else logits["count"][..., 0])
            metrics = {"frames": frames, "items": labels["count"][..., 0].numel(), "loss": loss.item() * frames,
                       **{f"{part}_loss": parts[part] * frames for part in ("stage", "count", "progress")},
                       "stage_acc": (logits["stage"].argmax(-1) == labels["stage"]).sum().item(),
                       "count_acc": hits.sum().item(),
                       "items_n2plus": (truth >= 2).sum().item(),
                       "count_acc_n2plus": hits[truth >= 2].sum().item(), "count_zero_acc": (truth == 0).sum().item(),
                       "progress_mae": (p - labels["progress"]).abs().sum().item(),
                       "episodes": 1, "count_final_acc": hits[-1].all().item(),
                       "synced": score is not None, "count_sync": score or 0.0,
                       **{f"items_n{b}": (buckets == b).sum().item() for b in range(BUCKETS)},
                       **{f"count_acc_n{b}": hits[buckets == b].sum().item() for b in range(BUCKETS)}}
            for key in ((name, sample["task"]), (name, None)):
                for metric, value in metrics.items():
                    sums[key][metric] += value
    model.train(was_training)

    def average(total):
        return {metric: total[metric] / total[DENOMINATOR.get(metric, "frames")] for metric in METRICS
                if total[DENOMINATOR.get(metric, "frames")]}

    overall, per_task = {}, defaultdict(dict)
    for (name, task), total in sums.items():
        for metric, value in average(total).items():
            if task is None:
                overall[f"val/{name}/{metric}"] = value
            else:
                per_task[task][f"{name}/{metric}"] = value
    result = {"overall": overall, "per_task": dict(sorted(per_task.items())), "episodes": count}
    if cfg.press_eval:
        result["press_test"] = press_test(model, dataset, cfg)
        result["overall"].update(result["press_test"]["overall"])
    return result


def take(sample, order):
    """按帧下标序列重排一条验证样本的帧和逐帧标签。"""
    index = torch.tensor(order)
    per_frame = ("stage", "count", "progress", "fed", "stage_mask", "count_mask", "event_mask", "keyframes",
                 "extra_keyframes")
    return {**sample, "frames": [sample["frames"][i] for i in order], **{key: sample[key][index] for key in per_frame}}


def post_press_events(truth, predicted, window=15):
    """每个按键第 k 次完成后 0–14 步的准确率、首次数对的滞后；遇到该键下一次完成则截断。

    未在窗口内数对的 latency 为 None，汇总时同时报告失败率，避免只看成功样本的滞后。
    """
    truth, predicted = truth.cpu().long(), predicted.cpu().long()
    events = []
    for item in range(truth.shape[1]):
        changes = ((truth[1:, item] > truth[:-1, item]).nonzero().flatten() + 1).tolist()
        for pos, start in enumerate(changes):
            end = min(start + window, changes[pos + 1] if pos + 1 < len(changes) else len(truth))
            hit = predicted[start:end, item] == truth[start:end, item]
            matched = hit.nonzero().flatten().tolist()
            events.append({"item": item, "k": truth[start, item].item(), "start": start,
                           "frames": len(hit), "correct": hit.sum().item(),
                           "latency": matched[0] if matched else None})
    return events


def press_summary(events):
    """只合并左键和中键，排除只需一次的确认键；每项每个 k 另行保留。"""
    events = [event for event in events if event["item"] < 2]
    if not events:
        return {}
    frames = sum(event["frames"] for event in events)
    latencies = [event["latency"] for event in events if event["latency"] is not None]
    return {"post_press_acc": sum(event["correct"] for event in events) / frames,
            "post_press_error": 1 - sum(event["correct"] for event in events) / frames,
            "latency_median": median(latencies) if latencies else None,
            "miss_rate": 1 - len(latencies) / len(events), "events": len(events), "frames": frames}


@torch.no_grad()
def press_test(model, dataset, cfg):
    """全部 press_button episode，原序和每条两种固定不规则间隔；与训练抽样和 eval_episodes 无关。

    不规则间隔沿用诊断脚本：必做一次局部 2x/0.5x 变速和 1–3 处停顿，无倒放。
    种子由 cfg.seed、episode 号和 variant 决定；帧、计数、stage、p、掩码一起重排，示范保持不变。
    best 可选 val/press_irregular/post_press_error（越低越好），原序分数只作为对照。
    """
    from qwen_training.progress.data import warp

    was_training = model.training
    model.eval()
    runs, groups = [], defaultdict(list)
    warp_cfg = replace(cfg, speed_prob=1.0, pause_prob=1.0, rewind_prob=0.0)
    try:
        for index, (task, episode) in enumerate(dataset.episodes):
            if task != "press_button" and getattr(dataset, "task_metadata", {}).get(task, {}).get("source_task") \
                    != "press_button":
                continue
            sample = dataset.sample(index, cfg)
            variants = [("normal", list(range(len(sample["frames"]))))]
            variants += [(f"irregular{v}", warp(len(sample["frames"]), warp_cfg,
                                                random.Random(cfg.seed * 100000 + episode * 1000 + v)))
                         for v in range(2)]
            for name, order in variants:
                changed = take(sample, order)
                with autocast(cfg):
                    outputs = model.rollout(changed, keyframes=inference_keyframes(cfg))
                pred = predicted_counts(model, outputs).cpu()
                truth = changed["count"].sum(-1).cpu()
                p = (F.softmax(torch.stack([o["progress"] for o in outputs]).float(), -1) * model.bins).sum(-1).cpu()
                events = post_press_events(truth, pred)
                run = {"episode": episode, "variant": name, "order": order, "events": events,
                       "truth": truth.long().tolist(), "prediction": pred.tolist(),
                       "count_acc": (pred == truth).float().mean().item(),
                       "progress_mae": (p - changed["progress"].cpu()).abs().mean().item()}
                runs.append(run)
                groups["normal" if name == "normal" else "irregular"].append(run)
    finally:
        model.train(was_training)
    overall, by_item_k = {}, {}
    for name, group in groups.items():
        events = [event for run in group for event in run["events"]]
        summary = press_summary(events)
        # 逐轨迹帧加权；各轨迹均为三个按键。
        frames = sum(len(run["order"]) for run in group)
        summary.update({key: sum(run[key] * len(run["order"]) for run in group) / frames
                        for key in ("count_acc", "progress_mae")})
        overall.update({f"val/press_{name}/{key}": value for key, value in summary.items() if value is not None})
        buckets = defaultdict(list)
        for event in events:
            if event["item"] < 2:
                buckets[f"{'left' if event['item'] == 0 else 'middle'}/k{event['k']}"].append(event)
        by_item_k[name] = {key: press_summary(bucket) for key, bucket in sorted(buckets.items())}
    return {"overall": overall, "by_item_k": by_item_k, "runs": runs, "seed": cfg.seed,
            "window": 15, "variants": 2, "keyframes": inference_keyframes(cfg)}


def pause_order(sample, length=PAUSE):
    """计数变化之间的中点插入 length 个相同帧，返回原步序列与停顿区间（含首尾）。"""
    changes = ((sample["count"][1:] != sample["count"][:-1]).flatten(1).any(-1).nonzero().flatten() + 1).tolist()
    pauses = sorted({(a + b) // 2 for a, b in zip([0, *changes], changes, strict=False) if b - a >= 2})
    order, segments = [], []
    for t in range(len(sample["frames"])):
        if t in pauses:
            segments.append({"source_step": t, "start": len(order), "end": len(order) + length})
        order += [t] * (length + 1 if t in pauses else 1)
    return order, segments


def pause_drifts(expected, progress, segments):
    """逐停顿段的原始预测漂移；与可视化共用，次数为期望次数、p 为 0-1 比例。"""
    result = []
    for segment in segments:
        start, end = segment["start"], segment["end"]
        seconds = (end - start) / SAMPLE_FPS
        count, p = (expected[end] - expected[start]).item(), (progress[end] - progress[start]).item()
        result.append({**segment, "count_drift": count, "p_drift": p,
                       "count_rate": count / seconds, "p_rate": p / seconds})
    return result


@torch.no_grad()
def pause_test(model, dataset, cfg, episodes=0, length=PAUSE, keyframes=None):
    """停顿测试：在每两次计数变化之间（及第一次之前）的中点，把那一帧重复 length 步，按 checkpoint 的推理模式运行。
    keyframes 省略时 none 配置保持无关键帧；其他配置使用模型触发，也可显式覆盖为 label/model/none。

    停顿期间画面不变、标签不变，看计数和 p 是否仍随步数上涨。
    count_drift：停顿结束比开始多出的期望次数（各项 Σ_r P(n_j ≥ r) 之和；event 为各项预测次数之和）；p_drift：p 的变化。
    按秒折算：count_rate、p_rate = 漂移 / 停顿秒数（5 Hz）。都接近 0 才说明不靠时间计数。
    """
    keyframes = keyframes or inference_keyframes(cfg)
    was_training = model.training
    model.eval()
    count = len(dataset) if episodes <= 0 else min(episodes, len(dataset))
    sums = defaultdict(lambda: defaultdict(float))
    for index in [i * len(dataset) // count for i in range(count)]:
        sample = dataset.sample(index, cfg)
        order, segments = pause_order(sample, length)
        with autocast(cfg):
            outputs = model.rollout(take(sample, order), keyframes=keyframes)
        expected = predicted_counts(model, outputs).sum(-1).float() if model.event \
            else torch.stack([out["count"] for out in outputs]).float().sigmoid().sum((-2, -1))
        p = (F.softmax(torch.stack([out["progress"] for out in outputs]).float(), -1) * model.bins).sum(-1)
        for segment in pause_drifts(expected, p, segments):
            for key in (sample["task"], None):
                sums[key]["pauses"] += 1
                sums[key]["count_drift"] += segment["count_drift"]
                sums[key]["p_drift"] += segment["p_drift"]
    model.train(was_training)

    def average(total):
        drift = {key: total[key] / total["pauses"] for key in ("count_drift", "p_drift")}
        return {**drift, "count_rate": drift["count_drift"] / (length / SAMPLE_FPS),
                "p_rate": drift["p_drift"] / (length / SAMPLE_FPS),
                "pauses": int(total["pauses"])}

    overall = average(sums.pop(None))
    return {"overall": overall, "per_task": {task: average(total) for task, total in sorted(sums.items())},
            "length": length, "keyframes": keyframes}


def main(argv=None):
    from qwen_training.progress.config import checkpoint_config
    from qwen_training.progress.data import RMBench
    from qwen_training.progress.model import ProgressModel

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--episodes", type=int, default=0, help="0 为该划分的全部 episode")
    # 带 --no- 的开关省略时沿用 checkpoint 的设置
    parser.add_argument("--count-feedback", action=argparse.BooleanOptionalAction,
                        help="--no-count-feedback 将回传概率置零，保留步骤信息与回传 MLP")
    parser.add_argument("--oracle-count-feedback", action=argparse.BooleanOptionalAction,
                        help="推理上限：回传上一帧真值累积概率，首步全零")
    parser.add_argument("--progress-reads-count", "--p-reads-count", action=argparse.BooleanOptionalAction,
                        help="--no-progress-reads-count 屏蔽 P 对 CNT token 的直接注意力")
    parser.add_argument("--count-reads-history", action=argparse.BooleanOptionalAction,
                        help="CNT 是否看之前的 CNT")
    parser.add_argument("--stage-reads-history", action=argparse.BooleanOptionalAction,
                        help="STG 是否看之前的 STG")
    parser.add_argument("--progress-reads-history", "--p-reads-history", action=argparse.BooleanOptionalAction,
                        help="P 是否看之前的 STG/CNT/P")
    parser.add_argument("--keyframes", choices=("model", "label", "none"),
                        help="指定单一关键帧模式；省略时 none 配置只验证 none，其他配置比较 oracle/model")
    parser.add_argument("--out", help="结果 JSON，默认写在 checkpoint 旁边")
    parser.add_argument("--pause", action="store_true", help="另做停顿测试（见 pause_test），结果在 \"pause\" 里")
    parser.add_argument("--press-test", action=argparse.BooleanOptionalAction,
                        help="全部 press_button 的原序 / 固定不规则间隔与逐 k 指标")
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint_config(checkpoint["config"], count_feedback=args.count_feedback,
                            oracle_count_feedback=args.oracle_count_feedback,
                            progress_reads_count=args.progress_reads_count, count_reads_history=args.count_reads_history,
                            stage_reads_history=args.stage_reads_history,
                            progress_reads_history=args.progress_reads_history, press_eval=args.press_test)
    model = ProgressModel.from_pretrained(cfg.model, cfg, cfg.device)
    model.load_trainable(checkpoint["model"])
    tasks = cfg.tasks.split(",") if cfg.tasks else None
    dataset = RMBench(cfg.labels, cfg.data, args.split, tasks, cfg.suites.split(",") if cfg.suites else None)
    result = evaluate(model, dataset, cfg, args.episodes, keyframes=args.keyframes)
    if args.pause:
        result["pause"] = pause_test(model, dataset, cfg, args.episodes, keyframes=args.keyframes)
    result.update(checkpoint=str(args.checkpoint), iteration=checkpoint["iteration"], split=args.split,
                  count_feedback=cfg.count_feedback, progress_reads_count=cfg.progress_reads_count,
                  count_reads_history=cfg.count_reads_history, stage_reads_history=cfg.stage_reads_history,
                  progress_reads_history=cfg.progress_reads_history,
                  oracle_count_feedback=cfg.oracle_count_feedback, keyframes=args.keyframes)
    out = Path(args.out or Path(args.checkpoint).with_name(f"{Path(args.checkpoint).stem}_{args.split}_eval.json"))
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result["overall"], **({"pause": result["pause"]["overall"]} if args.pause else {})}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
