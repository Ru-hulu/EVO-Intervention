# ruff: noqa: RUF002, RUF003
"""单卡训练：只有 L_P，整条轨迹切段后全部反传，段之间切断梯度。

python -m qwen_training.progress_lite.train --config qwen_training/progress_lite/configs/huawei_proprimem.yaml

输出目录（--out）：config.yaml、log.jsonl、latest.pt、best.pt（0.5、2 倍速的平均 MAE 最低）、step<N>.pt。
"""

import json
import math
import random
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qwen_training.progress_lite.clock_test import evaluate
from qwen_training.progress_lite.config import parse_config
from qwen_training.progress_lite.data import EpochSamples, Trajectories
from qwen_training.progress_lite.model import LoRALinear, ProgressLite, autocast, resize_images


def prepare_frames(sample, width, height):
    # Resize in loader workers before transferring the trajectory to the training process.
    frames = sample["frames"]
    unique = {id(image): image for views in frames for image in views}
    resized = dict(zip(unique, resize_images(unique.values(), width, height), strict=True))
    return {**sample, "frames": [[resized[id(image)] for image in views] for views in frames]}


def train(model, cfg, train_set, val_set):
    import yaml

    samples_per_epoch = len(EpochSamples(train_set, cfg, 0))
    steps_per_epoch = math.ceil(samples_per_epoch / cfg.accumulate)
    total = cfg.epochs * steps_per_epoch
    print(f"{samples_per_epoch} samples/epoch, {steps_per_epoch} updates/epoch, {total} updates total", flush=True)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True))

    lora = [
        param
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for param in (module.down.weight, module.up.weight)
    ]
    heads = [model.token, *model.head.parameters()]
    groups = {"head": (heads, cfg.head_lr), "lora": (lora, cfg.lr), "gate": (model.gates, cfg.gate_lr)}
    optimizer = torch.optim.Adam([{"params": params, "lr": lr, "name": name} for name, (params, lr) in groups.items()])

    def schedule(step):  # 线性预热，之后余弦衰减到 min_lr_ratio
        if step < cfg.warmup:
            return (step + 1) / cfg.warmup
        progress = (step - cfg.warmup) / max(1, total - cfg.warmup)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, best = 0, math.inf
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model.load_trainable(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["iteration"], checkpoint["best"]
        print(f"resumed from {cfg.resume} at iteration {start}", flush=True)

    def save(name, iteration):
        torch.save(
            {
                "config": asdict(cfg),
                "iteration": iteration,
                "best": best,
                "model": model.trainable_state(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            out / name,
        )

    def log(record):
        print(json.dumps(record), flush=True)
        with (out / "log.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    device = torch.device(cfg.device)
    model.train()
    loader, loaded_epoch = None, None
    for iteration in range(start + 1, total + 1):
        began, frames, total_loss = time.time(), 0, 0.0
        epoch, step = divmod(iteration - 1, steps_per_epoch)
        offset = step * cfg.accumulate
        if epoch != loaded_epoch:
            loader = iter(
                DataLoader(
                    EpochSamples(train_set, cfg, epoch, offset),
                    batch_size=None,
                    collate_fn=partial(prepare_frames, width=cfg.image_max_width, height=cfg.image_max_height),
                    num_workers=cfg.workers,
                    prefetch_factor=2 if cfg.workers else None,
                )
            )
            loaded_epoch = epoch
        trajectories = min(cfg.accumulate, samples_per_epoch - offset)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(trajectories):
            sample = next(loader)
            with autocast(cfg):
                for logits, begin, end in model.rollout_segments(sample):
                    # 按帧数加权：每条轨迹的权重相同，段长不同也不影响每帧的权重
                    weight = (end - begin) / (trajectories * len(sample["frames"]))
                    loss = weight * model.loss(logits, sample["progress"][begin:end].to(device))
                    loss.backward()
                    total_loss += loss.item()
            frames += len(sample["frames"])
        norm = torch.nn.utils.clip_grad_norm_(heads + lora + model.gates, cfg.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        seconds = time.time() - began
        log(
            {
                "iteration": iteration,
                "epoch": epoch + 1,
                "loss": total_loss,
                "grad_norm": norm,
                "seconds": round(seconds, 2),
                "frames_per_second": round(frames / seconds, 1),
                **(
                    {"peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                    if device.type == "cuda"
                    else {}
                ),
            }
        )
        if iteration % cfg.eval_every == 0 or iteration == total:
            metrics = evaluate(model, val_set, cfg, cfg.eval_episodes)
            log({"iteration": iteration, **metrics})
            if metrics["val/mae_speed"] < best:
                best = metrics["val/mae_speed"]
                save("best.pt", iteration)
        if iteration % cfg.save_every == 0 or iteration == total:
            save("latest.pt", iteration)
            save(f"step{iteration}.pt", iteration)


def main(argv=None):
    cfg = parse_config(argv, __doc__.splitlines()[0])
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = ProgressLite.from_pretrained(cfg.model, cfg, cfg.device)
    suites = cfg.suites.split(",") if cfg.suites else None
    train(
        model,
        cfg,
        Trajectories(cfg.labels, cfg.data, "train", suites),
        Trajectories(cfg.labels, cfg.data, "val", suites),
    )


if __name__ == "__main__":
    main()
