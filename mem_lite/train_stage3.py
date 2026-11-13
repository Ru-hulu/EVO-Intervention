"""三视角 + StreamForest 缓存，只训练语言 LoRA、历史投影和进度头。"""

import argparse
import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from time import time

import torch
import torch.distributed as dist
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.memory.vision import verify_vision_source
from mem_lite.memory.splitting import SPLIT_METHODS, validate_history_source
from mem_lite.models.m3_predictor import M3ProgressPredictor

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "stage3_train.json"
CHECKPOINT_FORMAT = "mem-lite-streamforest-v1"


def trainable_state(model):
    return {
        "m3_lora": get_peft_model_state_dict(model.qwen, save_embedding_layers=False),
        "history_projection": model.history_projection.state_dict(),
        "progress_head": model.progress_head.state_dict(),
    }


def load_trainable_state(model, state):
    set_peft_model_state_dict(model.qwen, state["m3_lora"])
    model.history_projection.load_state_dict(state["history_projection"])
    model.progress_head.load_state_dict(state["progress_head"])


def save_checkpoint(model, optimizer, config, output_dir, epoch, next_batch, step, epoch_finished=False,
                    scheduler=None):
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "python": random.getstate()}
    rng_by_rank = [None] * dist.get_world_size() if dist.is_initialized() else [rng]
    if dist.is_initialized():
        dist.all_gather_object(rng_by_rank, rng)
        if dist.get_rank() != 0:
            return
    state = {
        "format": CHECKPOINT_FORMAT, "model": trainable_state(model), "optimizer": optimizer.state_dict(),
        "config": config, "epoch": epoch, "next_batch": next_batch, "step": step, "rng_by_rank": rng_by_rank,
        "scheduler": None if scheduler is None else scheduler.state_dict(),
    }
    archive = epoch_finished and not config.get("keep_only_latest", False)
    names = ["last.pt", f"epoch-{epoch:02d}.pt"] if archive else ["last.pt"]
    for name in names:
        temporary = output_dir / f"{name}.tmp"
        torch.save(state, temporary)
        temporary.replace(output_dir / name)


def build_scheduler(optimizer, config, batches_per_epoch):
    """Optional warmup/cosine; older constant-LR configurations stay unchanged."""
    if config.get("lr_schedule", "constant") == "constant":
        return None
    if config["lr_schedule"] != "cosine" or not 0 <= config.get("warmup_ratio", 0) < 1:
        raise ValueError("Unsupported learning-rate schedule")
    from transformers import get_cosine_schedule_with_warmup
    steps = math.ceil(batches_per_epoch / config["gradient_accumulation_steps"]) * config["epochs"]
    config["total_optimizer_steps"] = steps
    config["warmup_steps"] = math.ceil(steps * config["warmup_ratio"])
    return get_cosine_schedule_with_warmup(optimizer, config["warmup_steps"], steps)


def initialization_audit(model, config, rank, world_size, checkpoint):
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    allowed = ("qwen.model.language_model.layers.", "history_projection.", "progress_head.")
    if any(not name.startswith(allowed) for name in names):
        raise ValueError("Unexpected trainable parameter outside native FoREST modules")
    lora_b = [p for name, p in model.named_parameters() if "lora_B" in name]
    fresh = bool(lora_b) and all(torch.count_nonzero(p).item() == 0 for p in lora_b)
    if checkpoint is None and not fresh:
        raise ValueError("Fresh initialization requires zero-initialized language LoRA B")
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            digest.update(name.encode())
            digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    audit = {"rank": rank, "world_size": world_size, "base_model": config["model_path"],
             "base_revision": config.get("base_revision"), "fresh_language_lora": fresh,
             "resume": checkpoint is not None, "initial_trainable_sha256": digest.hexdigest(),
             "trainable_parameter_names": names, "history_source": config["history_source"],
             "seed": config["seed"], "timestamp": time()}
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    suffix = "resume" if checkpoint is not None else "fresh"
    (output / f"initialization-rank{rank}-{suffix}.json").write_text(json.dumps(audit, indent=2) + "\n")


def read_config(args, checkpoint=None):
    config = (dict(checkpoint["config"]) if checkpoint is not None
              else json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("model_path", "sample_files", "output_dir", "epochs", "max_steps"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    method = getattr(args, "event_split_method", None)
    if method is not None:
        config["event_split_method"] = method
    if not config["model_path"] or not config["sample_files"]:
        raise ValueError("需要本地 Qwen3.5 基座及已构造的 StreamForest 样本")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
                "log_every_steps", "max_grad_norm"):
        if config[key] <= 0:
            raise ValueError(f"{key} 必须大于0")
    for key in ("save_every_steps", "save_every_epochs"):
        value = config.get(key, 1)
        if value is not None and value < 1:
            raise ValueError(f"{key} 必须为正整数或 null")
    if config["max_steps"] is not None and config["max_steps"] < 1:
        raise ValueError("max_steps 必须为正整数或 null")
    for key in ("model_path", "output_dir"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    config["sample_files"] = [str(Path(path).expanduser().resolve()) for path in config["sample_files"]]
    return config


def checkpoint_schedule(config, step, completed_epoch, epoch_finished, stop=False):
    """步数快照与 epoch 存档分别控制；提前到达 max_steps 时仍保留最后状态。"""
    steps, epochs = config["save_every_steps"], config.get("save_every_epochs", 1)
    archive = bool(epoch_finished and epochs is not None and completed_epoch % epochs == 0)
    due = bool(stop or archive or (steps is not None and step % steps == 0))
    return due, archive


def normalize_gradients(parameters, total_samples, world_size):
    """抵消 DDP 的 rank 平均，再按整个累积步的实际样本数归一化。"""
    scale = world_size / total_samples
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def configure_history(config, dataset):
    """从缓存元数据确定输入维度和容量，并随训练检查点保存。"""
    configured = config.get("history_capacity")
    if configured is not None and configured != dataset.history_capacity:
        raise ValueError("history_capacity 与样本记忆预算不一致")
    method = validate_history_source(dataset.history_source, config.get("event_split_method"))
    if config.get("history_source") is not None and config["history_source"] != dataset.history_source:
        raise ValueError("history_source 与样本不一致，不能覆盖已有检查点的记忆规则")
    config["event_split_method"] = method
    config["model"]["history_dim"] = dataset.history_dim
    config["history_capacity"] = dataset.history_capacity
    config["history_source"] = dataset.history_source


def train(config, checkpoint=None):
    if not torch.cuda.is_available():
        raise RuntimeError("第三阶段训练需要 CUDA；本地仅开发及离线检查，不应调用 train")
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if config.get("cuda_memory_fraction") is not None:
        fraction = float(config["cuda_memory_fraction"])
        if not 0 < fraction <= 1:
            raise ValueError("cuda_memory_fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    if world_size > 1:
        dist.init_process_group("nccl")
    try:
        _train(config, checkpoint, rank, world_size, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(config, checkpoint, rank, world_size, device):
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("第三阶段训练需要支持 BF16 的 GPU")
    if checkpoint is not None and len(checkpoint["rng_by_rank"]) != world_size:
        raise ValueError("断点续训需保持 GPU 数量，以恢复相同的样本顺序及梯度累积")
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    processor = AutoProcessor.from_pretrained(
        config["model_path"], max_pixels=config["max_pixels"], local_files_only=True,
    )
    dataset = M3Dataset(config["sample_files"], history_dim=config["model"]["history_dim"])
    if checkpoint is not None and config["data_signature"] != dataset.signature:
        raise ValueError("M3 样本或历史缓存已改变；续训必须使用相同的数据及顺序")
    config["data_signature"] = dataset.signature
    configure_history(config, dataset)
    verify_vision_source(config["model_path"], dataset.history_source)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=config["seed"], shuffle=True)
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], sampler=sampler,
        collate_fn=Stage3Collator(processor, dataset.history_dim, config["max_sequence_length"],
                                 config["history_capacity"]),
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config.get("persistent_workers", False) and config["num_workers"] > 0,
        multiprocessing_context="spawn" if config["num_workers"] > 0 else None,
        generator=torch.Generator().manual_seed(config["seed"] + rank),
    )
    model = M3ProgressPredictor(config["model_path"], **config["model"]).to(device)
    if dataset.history_dim != model.qwen.config.vision_config.out_hidden_size:
        raise ValueError("缓存的视觉特征维度与 Qwen 基座不一致")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"],
                                  fused=config.get("fused_optimizer", False))
    scheduler = build_scheduler(optimizer, config, len(loader))
    epoch, next_batch, step = 0, 0, 0
    if checkpoint is not None:
        load_trainable_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            if checkpoint.get("scheduler") is None:
                raise ValueError("Scheduled training cannot resume without its scheduler state")
            scheduler.load_state_dict(checkpoint["scheduler"])
        epoch, next_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"]
    if config.get("audit_initialization", False):
        initialization_audit(model, config, rank, world_size, checkpoint)
    # 视频首帧没有历史，某个 rank/微批次可能不使用 history_projection。
    wrapped = (DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                       find_unused_parameters=True) if world_size > 1 else model)
    if checkpoint is not None:
        rng = checkpoint["rng_by_rank"][rank]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        if "python" in rng:
            random.setstate(rng["python"])
        del checkpoint
    else:
        torch.manual_seed(config["seed"] + rank)
        torch.cuda.manual_seed(config["seed"] + rank)
    output_dir = Path(config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"kind": "start", "world_size": world_size, "samples": len(dataset),
                          "start_step": step, "trainable_parameters": sum(p.numel() for p in parameters)}), flush=True)
    if world_size > 1:
        dist.barrier()
    accumulation, max_steps = config["gradient_accumulation_steps"], config["max_steps"]
    wrapped.train()
    optimizer.zero_grad(set_to_none=True)
    while epoch < config["epochs"] and (max_steps is None or step < max_steps):
        sampler.set_epoch(epoch)
        totals = torch.zeros(5, device=device, dtype=torch.float64)  # CE和、样本数、绝对/平方/带符号误差和。
        microbatches, started = 0, perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index < next_batch:
                started = perf_counter()
                continue
            inputs = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                      for key, value in batch["inputs"].items()}
            target_pt = batch["target_pt"].to(device, non_blocking=True)
            microbatches += 1
            epoch_finished = batch_index + 1 == len(loader)
            update = microbatches == accumulation or epoch_finished
            synchronization = wrapped.no_sync() if world_size > 1 and not update else nullcontext()
            with synchronization:
                result = wrapped(inputs, target_pt)
                result["loss_sum"].backward()
            error = (result["pt"].detach() - target_pt).double()
            totals[0] += result["loss_sum"].detach().double()
            totals[1] += result["num_samples"]
            totals[2] += error.abs().sum()
            totals[3] += error.square().sum()
            totals[4] += error.sum()
            if not update:
                continue
            if world_size > 1:
                dist.all_reduce(totals)
            normalize_gradients(parameters, totals[1], world_size)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            next_epoch, cursor = (epoch + 1, 0) if epoch_finished else (epoch, batch_index + 1)
            stop = max_steps is not None and step >= max_steps
            log_due = step == 1 or step % config["log_every_steps"] == 0 or epoch_finished or stop
            if log_due:
                memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                                       torch.cuda.max_memory_reserved(device) / 2**30], device=device)
                memory_by_rank = [torch.empty_like(memory) for _ in range(world_size)]
                if world_size > 1:
                    dist.all_gather(memory_by_rank, memory)
                else:
                    memory_by_rank = [memory]
            if rank == 0 and log_due:
                torch.cuda.synchronize(device)
                seconds = perf_counter() - started
                record = {
                    "kind": "train", "epoch": epoch + 1, "step": step,
                    "loss": (totals[0] / totals[1]).item(), "samples": int(totals[1].item()),
                    "pt_mae_pp": (100 * totals[2] / totals[1]).item(),
                    "pt_rmse_pp": (100 * (totals[3] / totals[1]).sqrt()).item(),
                    "pt_bias_pp": (100 * totals[4] / totals[1]).item(),
                    "samples_per_second": totals[1].item() / seconds,
                    "step_seconds": seconds, "grad_norm": norm.item(),
                    "peak_allocated_gib_per_gpu": [float(v[0]) for v in memory_by_rank],
                    "peak_reserved_gib_per_gpu": [float(v[1]) for v in memory_by_rank],
                    "next_learning_rate": optimizer.param_groups[0]["lr"],
                    "timestamp": time(),
                }
                print(json.dumps(record), flush=True)
                with (output_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            save, archive = checkpoint_schedule(config, step, next_epoch, epoch_finished, stop)
            if save:
                save_checkpoint(model, optimizer, config, output_dir, next_epoch, cursor, step, archive,
                                scheduler=scheduler)
                if world_size > 1:
                    dist.barrier()
            totals.zero_()
            microbatches, started = 0, perf_counter()
            if stop:
                return
        epoch += 1
        next_batch = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", help="原始本地 Qwen3.5 基座")
    parser.add_argument("--sample-files", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="第三阶段自己的 last.pt 或 epoch-XX.pt")
    parser.add_argument("--event-split-method", choices=SPLIT_METHODS,
                        help="检查缓存是否使用指定切分方式；不会转换旧缓存")
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None and checkpoint.get("format") != CHECKPOINT_FORMAT:
        parser.error("--resume 只接受 StreamForest 进度模型检查点")
    train(read_config(args, checkpoint), checkpoint)


if __name__ == "__main__":
    main()
