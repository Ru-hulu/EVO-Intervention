# ruff: noqa: RUF002, RUF003
"""单卡训练：无梯度热身 + 短段反传（proposal_progress 5.1、5.2）。

两阶段分开运行：stage 1（--teacher true）[CNT] 回传真值计数、整段并行；stage 2 回传自己的预测、逐步计算，
用 --init-from 载入 stage 1 的 checkpoint。

python -m qwen_training.progress.train --config qwen_training/progress/configs/rmbench_2b_stage1.yaml [--key value ...]
python -m qwen_training.progress.train --config qwen_training/progress/configs/rmbench_2b_stage2.yaml \
    --init-from <stage 1 的 checkpoint>

输出目录（--out）：
- config.yaml：最终生效的配置；
- log.jsonl、tensorboard/：每次参数更新的损失、学习率、梯度范数、显存峰值、吞吐，以及验证指标；
- eval/iter<N>.json：每次验证的总体和逐任务指标；
- latest.pt、best.pt（按 best_metric）、step<N>.pt：可用 --resume 继续训练，或用 evaluate.py 单独评估。
"""

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from qwen_training.progress.config import parse_config
from qwen_training.progress.evaluate import evaluate
from qwen_training.progress.loss import autocast, sample_loss
from qwen_training.progress.model import LoRALinear


class TrainStream(IterableDataset):
    """无限地产生训练样本：随机 episode、随机起点；每个 worker 用不同的随机种子。"""

    def __init__(self, dataset, cfg, seed):
        self.dataset, self.cfg, self.seed = dataset, cfg, seed

    def __iter__(self):
        worker = get_worker_info()
        rng = random.Random(self.seed * 1000 + (worker.id if worker else 0))
        while True:
            yield self.dataset.sample(rng.randrange(len(self.dataset)), self.cfg, rng, self.cfg.segment)


def keep_sample(sample):
    return sample


def train(model, cfg, train_set, val_set=None):
    import yaml
    from torch.utils.tensorboard import SummaryWriter

    if cfg.resume and cfg.init_from:
        raise ValueError("resume and init_from are mutually exclusive")

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
        progress = (step - cfg.warmup) / max(1, cfg.iterations - cfg.warmup)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    start, best = 0, math.inf
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        model.load_trainable(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["iteration"], checkpoint["best"]
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
                    "model": {key: value.detach().cpu() for key, value in model.named_parameters()
                              if value.requires_grad},
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
        log({"iteration": iteration, **result["overall"], "eval_seconds": round(time.time() - began, 1)}, iteration)
        value = result["overall"].get(cfg.best_metric, math.inf)
        if value < best:
            best = value
            save("best.pt", iteration)

    # 后台进程读 HDF5、解码图像；断点续训时换一个种子，避免重复同一批样本。
    loader = iter(DataLoader(TrainStream(train_set, cfg, cfg.seed + start), batch_size=None, collate_fn=keep_sample,
                             num_workers=cfg.workers, prefetch_factor=2 if cfg.workers else None,
                             persistent_workers=cfg.workers > 0))
    cuda = torch.device(cfg.device).type == "cuda"
    if val_set is not None and cfg.eval_at_start and start == 0:
        validate(0, cfg.eval_episodes)
    model.train()
    history = []
    for iteration in range(start + 1, cfg.iterations + 1):
        began, record = time.time(), {}
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        frames = 0
        for _ in range(cfg.accumulate):
            sample = next(loader)
            with autocast(cfg):
                outputs = model.rollout(sample, sample["grad_from"], teacher=cfg.teacher)
                loss, parts = sample_loss(model, outputs, sample, cfg)
                loss = loss + cfg.gate_l2 * sum(((param - init) ** 2).sum() for param, init in model.gates)
            (loss / cfg.accumulate).backward()
            frames += len(sample["frames"])
            for name, value in {"loss": loss.item(), **parts}.items():
                record[name] = record.get(name, 0.0) + value / cfg.accumulate
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
        last = iteration == cfg.iterations
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
    model = ProgressModel.from_pretrained(cfg.model, cfg, cfg.device)
    train(model, cfg, RMBench(cfg.labels, cfg.data, "train", tasks), RMBench(cfg.labels, cfg.data, "val", tasks))


if __name__ == "__main__":
    main()
