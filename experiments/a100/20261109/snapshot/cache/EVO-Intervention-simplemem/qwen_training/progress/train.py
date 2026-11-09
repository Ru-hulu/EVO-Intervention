# ruff: noqa: RUF002, RUF003
"""单卡训练：无梯度热身 + 短段反传（proposal_progress 5.1、5.2）。

默认单阶段训练（--teacher false）：无反馈、无关键帧的 level 模式按 --chunk-size 并行，
视觉编码和输出头批量计算；--window 独立控制可见帧数。
--warmup-chunk-size / --train-chunk-size 可分别覆盖热身 / 反传的并行帧数；0 沿用 --chunk-size。
--teacher true 可选真值回传并行训练；--init-from 只载入权重，不要求先跑另一个阶段。

python -m qwen_training.progress.train --config qwen_training/progress/configs/autodl_rmbench_4b.yaml [--key value ...]

输出目录（--out）：
- config.yaml：最终生效的配置；
- log.jsonl、tensorboard/：每次参数更新的损失、学习率、梯度范数、显存峰值、吞吐，以及验证指标；
- eval/iter<N>.json：每次验证的总体和逐任务指标；
- latest.pt、best.pt（按 best_metric）、step<N>.pt：可用 --resume 继续训练，或用 evaluate.py 单独评估。
"""

import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from qwen_training.datasets.rmbench.spec import FIXED
from qwen_training.datasets.rmbench.tasks import TASKS
from qwen_training.progress.config import parse_config
from qwen_training.progress.evaluate import evaluate, take
from qwen_training.progress.loss import autocast, sample_loss
from qwen_training.progress.model import LoRALinear, Standardize


class TrainStream(IterableDataset):
    """无限地产生训练样本：随机 episode、随机起点；每个 worker 用不同的随机种子。

    repeat_boost > 0 时每个任务等概率，计数会超过 1 的任务（非固定流程）再乘 repeat_boost。
    count_balance 时先等概率抽计数等级，再从该等级的帧里抽一帧，带梯度的段包含它（优先于 repeat_boost）。
    """

    def __init__(self, dataset, cfg, seed):
        self.dataset, self.cfg, self.seed = dataset, cfg, seed
        self.levels = [level for level in dataset.count_levels() if level] if cfg.count_balance else None
        self.weights = None
        if cfg.repeat_boost > 0:
            tasks = Counter(task for task, _ in dataset.episodes)
            specs = getattr(dataset, "task_specs", None)
            specs = specs if isinstance(specs, dict) else TASKS
            self.weights = [(cfg.repeat_boost if specs[task].kind != FIXED else 1) / tasks[task]
                            for task, _ in dataset.episodes]

    def __iter__(self):
        worker = get_worker_info()
        rng = random.Random(self.seed * 1000 + (worker.id if worker else 0))
        while True:
            if self.levels:
                index, anchor = rng.choice(rng.choice(self.levels))
                yield self.dataset.sample(index, self.cfg, rng, self.cfg.segment, anchor)
                continue
            index = rng.choices(range(len(self.dataset)), self.weights)[0] if self.weights \
                else rng.randrange(len(self.dataset))
            yield self.dataset.sample(index, self.cfg, rng, self.cfg.segment)


def keep_sample(sample):
    return sample


class EpochSamples(Dataset):
    """A shuffled finite epoch, covering every episode at least once.

    With repeat_boost, task totals follow the old task-balanced weights, rounded
    to whole samples. Extra visits are spread evenly over each task's episodes.
    Per-position random seeds make augmentation independent of worker prefetch.
    """

    def __init__(self, dataset, cfg, epoch, offset=0):
        self.dataset, self.cfg, self.epoch, self.offset = dataset, cfg, epoch, offset
        groups = {}
        for index, (task, _) in enumerate(dataset.episodes):
            groups.setdefault(task, []).append(index)
        rng = random.Random(f"{cfg.seed}:epoch:{epoch}")
        largest = max(map(len, groups.values()))
        specs = getattr(dataset, "task_specs", None)
        specs = specs if isinstance(specs, dict) else TASKS
        order = []
        for task, indices in groups.items():
            target = len(indices)
            if cfg.repeat_boost > 0:
                boost = cfg.repeat_boost if specs[task].kind != FIXED else 1
                target = max(len(indices), round(largest * boost / min(1, cfg.repeat_boost)))
            whole, rest = divmod(target, len(indices))
            order.extend(indices * whole + rng.sample(indices, rest))
        rng.shuffle(order)
        self.order = order

    def __len__(self):
        return len(self.order) - self.offset

    def __getitem__(self, position):
        position += self.offset
        rng = random.Random(f"{self.cfg.seed}:epoch:{self.epoch}:sample:{position}")
        return self.dataset.sample(self.order[position], self.cfg, rng, self.cfg.segment)


def fit_count_norm(model, samples, cfg):
    """训练前把几条训练样本整条跑一遍（无梯度），汇总计数头输入在所有步、所有项上的一、二阶矩，设定标准化的统计量。"""
    moments = []

    def collect(module, args):
        x = args[0].detach().float().reshape(-1, args[0].shape[-1])
        moments.append((len(x), x.sum(0), x.pow(2).sum(0)))

    handle = model.count_norm.register_forward_pre_hook(collect)
    try:
        with torch.no_grad(), autocast(cfg):
            for sample in samples:
                model.rollout(sample, len(sample["frames"]), teacher=cfg.teacher)
    finally:
        handle.remove()
    rows = sum(n for n, _, _ in moments)
    model.count_norm.fit(sum(s for _, s, _ in moments) / rows, sum(q for _, _, q in moments) / rows)


def train(model, cfg, train_set, val_set=None):
    import yaml
    from torch.utils.tensorboard import SummaryWriter

    if cfg.resume and cfg.init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    if cfg.epochs < 0 or cfg.max_updates < 0 or cfg.accumulate < 1:
        raise ValueError("epochs/max_updates must be >= 0 and accumulate must be > 0")
    if cfg.epochs and cfg.count_balance:
        raise ValueError("epoch coverage is incompatible with global count_balance sampling")
    if cfg.segments_per_trajectory < 1 or cfg.accumulate % cfg.segments_per_trajectory:
        raise ValueError("accumulate must be divisible by segments_per_trajectory")
    if cfg.segments_per_trajectory > 1 and (not cfg.epochs or cfg.count_balance or cfg.teacher
            or cfg.keyframes != "none" or cfg.count_feedback or cfg.count_mode != "level"):
        raise ValueError("multi-segment training requires epoch sampling, no-feedback level mode, keyframes=none")
    trajectories_per_update = cfg.accumulate // cfg.segments_per_trajectory
    first_epoch = EpochSamples(train_set, cfg, 0) if cfg.epochs else None
    samples_per_epoch = len(first_epoch) if first_epoch is not None else 0
    steps_per_epoch = math.ceil(samples_per_epoch / trajectories_per_update) if cfg.epochs else 0
    total_iterations = cfg.epochs * steps_per_epoch if cfg.epochs else cfg.iterations
    if cfg.max_updates > 0:
        total_iterations = min(total_iterations, cfg.max_updates)
    if total_iterations < 1:
        raise ValueError("training budget must be > 0")
    signature = hashlib.sha256(json.dumps({
        "episodes": train_set.episodes, "samples": samples_per_epoch, "accumulate": cfg.accumulate,
        "seed": cfg.seed, "repeat_boost": cfg.repeat_boost, "segments_per_trajectory": cfg.segments_per_trajectory,
    }).encode()).hexdigest() if cfg.epochs else None
    if cfg.epochs:
        print(f"{cfg.epochs} epochs, {samples_per_epoch} samples/epoch, "
              f"{steps_per_epoch} updates/epoch, {total_iterations} updates total", flush=True)

    out = Path(cfg.out)
    (out / "eval").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True))
    writer = SummaryWriter(out / "tensorboard")

    # 三组参数：随机初始化的部分、LoRA、GDN 门。
    gates = [param for param, _ in model.gates]
    lora = [param for module in model.modules() if isinstance(module, LoRALinear)
            for param in (module.down.weight, module.up.weight)]
    taken = {id(param) for param in gates + lora}
    heads = [param for param in model.parameters() if param.requires_grad and id(param) not in taken]
    groups = {"head": (heads, cfg.head_lr), "lora": (lora, cfg.lr), "gate": (gates, cfg.gate_lr)}
    optimizer = torch.optim.AdamW([{"params": params, "lr": lr, "name": name} for name, (params, lr) in groups.items()],
                                  weight_decay=cfg.weight_decay)

    def schedule(step):   # 线性预热，之后余弦衰减到 min_lr_ratio
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        progress = (step - cfg.warmup) / max(1, total_iterations - cfg.warmup)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    start, best = 0, math.inf
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("epoch_signature") != signature:
            raise ValueError("resume sampling differs; use init_from for an old iteration checkpoint or changed dataset")
        model.load_trainable(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["iteration"], checkpoint["best"]
        if start > total_iterations:
            raise ValueError("resume iteration exceeds training budget")
        random.setstate(checkpoint["rng"]["python"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if checkpoint["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
        print(f"resumed from {cfg.resume} at iteration {start}", flush=True)
    else:
        if cfg.init_from:
            model.load_trainable(torch.load(cfg.init_from, map_location="cpu", weights_only=False)["model"])
            print(f"initialized from {cfg.init_from}", flush=True)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    def save(name, iteration):
        torch.save({"config": asdict(cfg), "iteration": iteration, "best": best,
                    "epoch_signature": signature,
                    "epoch": math.ceil(iteration / steps_per_epoch) if cfg.epochs else None,
                    "epoch_step": (iteration - 1) % steps_per_epoch + 1 if cfg.epochs and iteration else 0,
                    "model": model.trainable_state(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}},
                   out / name)

    def log(record, iteration):
        print(json.dumps(record), flush=True)
        with (out / "log.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        for key, value in record.items():
            if isinstance(value, (int, float)) and key != "iteration":
                writer.add_scalar(key if "/" in key else f"train/{key}", value, iteration)

    def validate(iteration, episodes):
        nonlocal best
        began = time.time()
        result = evaluate(model, val_set, cfg, episodes)
        (out / "eval" / f"iter{iteration}.json").write_text(json.dumps(result, indent=2) + "\n")
        log({"iteration": iteration, **({"epoch": math.ceil(iteration / steps_per_epoch)} if cfg.epochs else {}),
             **result["overall"], "eval_seconds": round(time.time() - began, 1)}, iteration)
        value = result["overall"].get(cfg.best_metric, math.inf)
        if value < best:
            best = value
            save("best.pt", iteration)

    # 后台进程读 HDF5、解码图像；断点续训时换一个种子，避免重复同一批样本。
    def make_loader(samples, epoch):
        return iter(DataLoader(samples, batch_size=None, collate_fn=keep_sample, num_workers=cfg.workers,
                               prefetch_factor=2 if cfg.workers else None, persistent_workers=cfg.workers > 0,
                               generator=torch.Generator().manual_seed(cfg.seed + epoch) if cfg.epochs else None))

    loader = None if cfg.epochs else make_loader(TrainStream(train_set, cfg, cfg.seed + start), start)
    cuda = torch.device(cfg.device).type == "cuda"
    if isinstance(model.count_norm, Standardize) and not model.count_norm.initialized:
        norm_samples = [first_epoch[i % samples_per_epoch] for i in range(cfg.count_norm_samples)] if cfg.epochs \
            else [next(loader) for _ in range(cfg.count_norm_samples)]
        fit_count_norm(model, norm_samples, cfg)
        print(f"count_norm fitted on {cfg.count_norm_samples} samples", flush=True)
    if val_set is not None and cfg.eval_at_start and start == 0:
        validate(0, cfg.eval_episodes)
    model.train()
    history = []
    loaded_epoch = None
    for iteration in range(start + 1, total_iterations + 1):
        began, record = time.time(), {}
        batch_trajectories = trajectories_per_update
        batch_accumulate = cfg.accumulate
        if cfg.epochs:
            epoch, epoch_step = divmod(iteration - 1, steps_per_epoch)
            offset = epoch_step * trajectories_per_update
            if epoch != loaded_epoch:
                loader = make_loader(EpochSamples(train_set, cfg, epoch, offset), epoch)
                loaded_epoch = epoch
            batch_trajectories = min(trajectories_per_update, samples_per_epoch - offset)
            batch_accumulate = batch_trajectories * cfg.segments_per_trajectory
            record.update(epoch=epoch + 1, epoch_step=epoch_step + 1, steps_per_epoch=steps_per_epoch,
                          epochs=cfg.epochs, samples=batch_trajectories, segments=batch_accumulate)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        frames = 0
        for _ in range(batch_trajectories):
            sample = next(loader)
            if cfg.segments_per_trajectory > 1:
                segments = model.rollout_segments(sample)
            else:
                def single_segment(sample=sample):
                    outputs = model.rollout(sample, sample["grad_from"], teacher=cfg.teacher)
                    yield outputs, sample["grad_from"], len(sample["frames"])
                segments = single_segment()
            with autocast(cfg):
                for outputs, lo, hi in segments:
                    labels = take(sample, list(range(lo, hi)))
                    labels["grad_from"] = 0
                    loss, parts = sample_loss(model, outputs, labels, cfg, cfg.count_margin, cfg.event_weight)
                    loss = loss + cfg.gate_l2 * sum(((param - init) ** 2).sum() for param, init in model.gates)
                    (loss / batch_accumulate).backward()
                    for name, value in {"loss": loss.item(), **parts}.items():
                        record[name] = record.get(name, 0.0) + value / batch_accumulate
                    del outputs, loss
            frames += len(sample["frames"])
        record["grad_norm"] = torch.nn.utils.clip_grad_norm_(heads + lora + gates, cfg.grad_clip).item()
        for group in optimizer.param_groups:
            record[f"lr_{group['name']}"] = group["lr"]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        seconds = time.time() - began
        record.update(iteration=iteration, seconds=round(seconds, 2), frames_per_second=round(frames / seconds, 1))
        if cuda:   # 这次参数更新期间（含热身、反传和优化器）实际分配的显存峰值
            record["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if iteration % cfg.log_every == 0:
            log(record, iteration)
        history.append(record)
        last = iteration == total_iterations
        if val_set is not None and (iteration % cfg.eval_every == 0 or last):
            validate(iteration, cfg.final_eval_episodes if last else cfg.eval_episodes)
        if iteration % cfg.save_every == 0 or last:
            save("latest.pt", iteration)
            save(f"step{iteration}.pt", iteration)
    writer.close()
    return history


def main(argv=None):
    from qwen_training.progress.data import RMBench
    from qwen_training.progress.model import ProgressModel

    cfg = parse_config(argv, __doc__.splitlines()[0])
    tasks = cfg.tasks.split(",") if cfg.tasks else None
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)   # 在构建模型之前设：新加部分（readout、MLP 头、LoRA）的随机初始化也由 seed 决定
    model = ProgressModel.from_pretrained(cfg.model, cfg, cfg.device)
    suites = cfg.suites.split(",") if cfg.suites else None
    val_set = RMBench(cfg.labels, cfg.data, "val", tasks, suites) if cfg.eval_every > 0 else None
    train(model, cfg, RMBench(cfg.labels, cfg.data, "train", tasks, suites), val_set)


if __name__ == "__main__":
    main()
