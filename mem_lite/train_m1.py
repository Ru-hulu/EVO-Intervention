"""M1 单卡 / torchrun 训练入口；仅语言层 LoRA，无验证或评测流程。"""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor

from mem_lite.datasets.dataset_manifest import load_sample_files
from mem_lite.datasets.m1_dataset import M1Collator, M1Dataset
from mem_lite.models.m1_segmenter import M1Segmenter

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "m1_train.json"


def save_checkpoint(model, optimizer, config, output_dir, epoch, next_batch, step, epoch_finished=False):
    """只在完整优化步后保存；所有 rank 参与收集 RNG，rank 0 写文件。"""
    distributed = dist.is_initialized()
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
    rng_by_rank = [None] * dist.get_world_size() if distributed else [rng]
    if distributed:
        dist.all_gather_object(rng_by_rank, rng)
        if dist.get_rank() != 0:
            return
    state = {
        "format": "mem-lite-m1-lora-v1",
        "lora": get_peft_model_state_dict(model.qwen, save_embedding_layers=False),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "next_batch": next_batch,
        "step": step,
        "rng_by_rank": rng_by_rank,
    }
    temporary = output_dir / "last.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(output_dir / "last.pt")
    if epoch_finished:
        temporary = output_dir / f"epoch-{epoch:02d}.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(output_dir / f"epoch-{epoch:02d}.pt")


def read_config(args, checkpoint):
    config = (dict(checkpoint["config"]) if checkpoint is not None
              else json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("model_path", "sample_files", "output_dir", "epochs", "max_steps"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    manifest = getattr(args, "data_manifest", None)
    if manifest is None and checkpoint is None:
        manifest = config.get("data_manifest")
    if manifest:
        if args.sample_files is not None:
            raise ValueError("data_manifest与sample_files不能同时指定")
        config["sample_files"] = load_sample_files(manifest, "m1", split="train")
        config["data_manifest"] = str(Path(manifest).expanduser().resolve())
    elif args.sample_files is not None:
        config.pop("data_manifest", None)
    if not config["model_path"] or not config["sample_files"]:
        raise ValueError("请在配置或命令行中提供本地 model_path 和 sample_files")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "log_every_steps", "save_every_steps"):
        if config[key] < 1:
            raise ValueError(f"{key} 必须大于 0")
    # 将相对数据路径固定下来，方便从同一检查点恢复。
    config["model_path"] = str(Path(config["model_path"]).expanduser().resolve())
    config["sample_files"] = [str(Path(path).expanduser().resolve()) for path in config["sample_files"]]
    config["output_dir"] = str(Path(config["output_dir"]).expanduser().resolve())
    return config


def train(config, checkpoint=None):
    if not torch.cuda.is_available():
        raise RuntimeError("训练入口需要 CUDA GPU；本地代码检查无需运行训练入口")
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl")
    try:
        _train(config, checkpoint, rank, world_size, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(config, checkpoint, rank, world_size, device):
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前训练配置使用 BF16，需要支持 BF16 的 GPU")
    if checkpoint is not None and len(checkpoint["rng_by_rank"]) != world_size:
        raise ValueError("断点续训需保持原 GPU 数量，以恢复相同的数据顺序和梯度累积")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    processor = AutoProcessor.from_pretrained(
        config["model_path"], max_pixels=config["max_pixels"], local_files_only=True,
    )
    dataset = M1Dataset(config["sample_files"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=config["seed"], shuffle=True)
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], sampler=sampler,
        collate_fn=M1Collator(processor, config["max_sequence_length"]),
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
        multiprocessing_context="spawn" if config["num_workers"] > 0 else None,
        generator=torch.Generator().manual_seed(config["seed"] + rank),
    )
    model = M1Segmenter(
        config["model_path"], lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], gradient_checkpointing=config["gradient_checkpointing"],
    ).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epoch, next_batch, step = 0, 0, 0
    if checkpoint is not None:
        set_peft_model_state_dict(model.qwen, checkpoint["lora"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch, next_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"]
    wrapped = (DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
               if world_size > 1 else model)
    if checkpoint is not None:
        rng = checkpoint["rng_by_rank"][rank]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        del checkpoint
    else:
        # DDP 初始化已同步参数；不同 rank 使用不同的 dropout 随机流。
        torch.manual_seed(config["seed"] + rank)
        torch.cuda.manual_seed(config["seed"] + rank)

    output_dir = Path(config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "kind": "start", "world_size": world_size, "samples": len(dataset),
            "trainable_parameters": sum(p.numel() for p in parameters), "start_step": step,
            "expert_tokens_enabled": False,
        }), flush=True)
    if world_size > 1:
        dist.barrier()

    accumulation = config["gradient_accumulation_steps"]
    max_steps = config["max_steps"]
    wrapped.train()
    optimizer.zero_grad(set_to_none=True)
    while epoch < config["epochs"] and (max_steps is None or step < max_steps):
        sampler.set_epoch(epoch)
        totals = torch.zeros(3, device=device, dtype=torch.float64)  # 损失和、答案 token 数、样本数。
        microbatches = 0
        started = perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index < next_batch:
                started = perf_counter()
                continue
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            microbatches += 1
            epoch_finished = batch_index + 1 == len(loader)
            update = microbatches == accumulation or epoch_finished
            synchronization = wrapped.no_sync() if world_size > 1 and not update else nullcontext()
            with synchronization:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = wrapped(batch)
                    loss_sum = result["loss_sum"]
                loss_sum.backward()
            totals[0] += loss_sum.detach().double()
            totals[1] += result["text_tokens"].detach()
            totals[2] += batch["input_ids"].shape[0]
            if not update:
                continue
            if world_size > 1:
                dist.all_reduce(totals)
            # 每个答案 token 等权；不能先平均每个微批次，再让不同长度的答案等权。
            scale = world_size / totals[1]
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            next_epoch, cursor = (epoch + 1, 0) if epoch_finished else (epoch, batch_index + 1)
            stop = max_steps is not None and step >= max_steps
            if rank == 0 and (step % config["log_every_steps"] == 0 or epoch_finished or stop):
                torch.cuda.synchronize(device)
                seconds = perf_counter() - started
                record = {
                    "kind": "train", "step": step, "epoch": epoch + 1,
                    "loss": (totals[0] / totals[1]).item(), "text_tokens": int(totals[1].item()),
                    "samples_per_second": totals[2].item() / seconds,
                    "step_seconds": seconds, "grad_norm": norm.item(),
                }
                print(json.dumps(record), flush=True)
                with (output_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            if step % config["save_every_steps"] == 0 or epoch_finished or stop:
                save_checkpoint(model, optimizer, config, output_dir, next_epoch, cursor, step, epoch_finished)
                if world_size > 1:
                    dist.barrier()
            totals.zero_()
            microbatches = 0
            started = perf_counter()
            if stop:
                return
        epoch += 1
        next_batch = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", help="本地 Qwen3.5-0.8B 权重目录，不自动下载")
    parser.add_argument("--sample-files", nargs="+", help="已构造的 samples.jsonl 或其所在目录")
    parser.add_argument("--data-manifest", help="批量数据清单，仅加载其中的train视频")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int, help="目标总优化步数，包含检查点中已完成的步数")
    parser.add_argument("--resume", help="本入口保存的 last.pt 或 epoch-XX.pt")
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None and checkpoint.get("format") != "mem-lite-m1-lora-v1":
        parser.error("仅支持 mem-lite M1 LoRA 检查点，旧进度模型检查点不能在此恢复")
    config = read_config(args, checkpoint)
    train(config, checkpoint)


if __name__ == "__main__":
    main()
