"""第二阶段联合训练入口：M1 LoRA + 事件编码器/M2/历史投影；没有验证或评测流程。"""

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
from mem_lite.datasets.m2_dataset import Stage2Collator, Stage2Dataset
from mem_lite.models.stage2_model import Stage2Model

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "stage2_train.json"
CHECKPOINT_FORMAT = "mem-lite-stage2-v1"


def trainable_state(model):
    return {
        "m1_lora": get_peft_model_state_dict(model.segmenter.qwen, save_embedding_layers=False),
        "event_encoder": model.event_encoder.state_dict(),
        "m2_compressor": model.compressor.state_dict(),
        "history_projection": model.decoder.history_projection.state_dict(),
    }


def load_trainable_state(model, state):
    set_peft_model_state_dict(model.segmenter.qwen, state["m1_lora"])
    model.event_encoder.load_state_dict(state["event_encoder"])
    model.compressor.load_state_dict(state["m2_compressor"])
    model.decoder.history_projection.load_state_dict(state["history_projection"])


def save_checkpoint(model, optimizer, config, output_dir, epoch, next_batch, step, epoch_finished=False):
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
    rng_by_rank = [None] * dist.get_world_size() if dist.is_initialized() else [rng]
    if dist.is_initialized():
        dist.all_gather_object(rng_by_rank, rng)
        if dist.get_rank() != 0:
            return
    state = {
        "format": CHECKPOINT_FORMAT, "model": trainable_state(model), "optimizer": optimizer.state_dict(),
        "config": config, "epoch": epoch, "next_batch": next_batch, "step": step, "rng_by_rank": rng_by_rank,
    }
    names = ["last.pt", f"epoch-{epoch:02d}.pt"] if epoch_finished else ["last.pt"]
    for name in names:
        temporary = output_dir / f"{name}.tmp"
        torch.save(state, temporary)
        temporary.replace(output_dir / name)


def read_config(args, checkpoint=None):
    config = (dict(checkpoint["config"]) if checkpoint is not None
              else json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("model_path", "m1_sample_files", "m2_sample_files", "output_dir", "epochs", "max_steps"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    manifest = getattr(args, "data_manifest", None)
    if manifest is None and checkpoint is None:
        manifest = config.get("data_manifest")
    if manifest:
        if args.m1_sample_files is not None or args.m2_sample_files is not None:
            raise ValueError("data_manifest与单独的m1/m2_sample_files不能同时指定")
        config["m1_sample_files"] = load_sample_files(manifest, "m1", split="train")
        config["m2_sample_files"] = load_sample_files(manifest, "m2", split="train")
        config["data_manifest"] = str(Path(manifest).expanduser().resolve())
    elif args.m1_sample_files is not None or args.m2_sample_files is not None:
        config.pop("data_manifest", None)
    if args.m1_checkpoint is not None:
        if checkpoint is not None:
            raise ValueError("--resume 已包含 M1 权重，不能再指定 --m1-checkpoint")
        config["m1_checkpoint"] = args.m1_checkpoint
    if not config["model_path"] or not config["m1_sample_files"] or not config["m2_sample_files"]:
        raise ValueError("需要原始本地模型目录、M1样本及M2队列样本")
    if checkpoint is None and not config["m1_checkpoint"]:
        raise ValueError("第二阶段需通过 m1_checkpoint 接续第一阶段，不能静默从随机 LoRA 开始")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "vision_batch_size",
                "log_every_steps", "save_every_steps", "m1_loss_weight", "memory_loss_weight"):
        if config[key] <= 0:
            raise ValueError(f"{key} 必须大于0")
    for key in ("model_path", "output_dir", "m1_checkpoint"):
        if config[key]:
            config[key] = str(Path(config[key]).expanduser().resolve())
    for key in ("m1_sample_files", "m2_sample_files"):
        config[key] = [str(Path(path).expanduser().resolve()) for path in config[key]]
    return config


def normalize_gradients(optimizer, totals, world_size, config):
    """DDP 已按 rank 平均梯度，各分支再按其全局答案 token 总量归一化。"""
    for group, token_index in zip(optimizer.param_groups, (1, 3), strict=True):
        scale = world_size * config[f"{group['name']}_loss_weight"] / totals[token_index]
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)


def train(config, checkpoint=None):
    if not torch.cuda.is_available():
        raise RuntimeError("第二阶段训练需要 CUDA；本地仅开发及离线检查，不应调用 train")
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
        raise RuntimeError("第二阶段训练需要支持 BF16 的 GPU")
    if checkpoint is not None and len(checkpoint["rng_by_rank"]) != world_size:
        raise ValueError("断点续训需保持 GPU 数量，以恢复相同的样本顺序及梯度累积")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    initial = None
    if checkpoint is None:
        initial = torch.load(config["m1_checkpoint"], map_location="cpu", weights_only=True)
        if initial.get("format") != "mem-lite-m1-lora-v1":
            raise ValueError("m1_checkpoint 必须来自 mem_lite.train_m1")
        config["m1"] = {key: initial["config"][key] for key in (
            "lora_rank", "lora_alpha", "lora_dropout", "gradient_checkpointing",
        )}
    processor = AutoProcessor.from_pretrained(
        config["model_path"], max_pixels=config["max_pixels"], local_files_only=True,
    )
    dataset = Stage2Dataset(config["m2_sample_files"], config["m1_sample_files"], config["seed"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=config["seed"], shuffle=True)
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], sampler=sampler,
        collate_fn=Stage2Collator(processor, config["max_sequence_length"]),
        num_workers=config["num_workers"], pin_memory=True,
        # 每轮重新创建 worker，使 dataset.epoch 生效；配对只依赖 seed/epoch/index。
        persistent_workers=False, multiprocessing_context="spawn" if config["num_workers"] > 0 else None,
        generator=torch.Generator().manual_seed(config["seed"] + rank),
    )
    model = Stage2Model(
        config["model_path"], config["m1"], config["event_encoder"], config["m2_compressor"],
        config["vision_batch_size"], config["max_text_sequence_length"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(config["m1_learning_rate"], config["memory_learning_rate"]),
        weight_decay=config["weight_decay"],
    )
    epoch, next_batch, step = 0, 0, 0
    if checkpoint is not None:
        load_trainable_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch, next_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"]
    else:
        set_peft_model_state_dict(model.segmenter.qwen, initial["lora"])
        del initial
    wrapped = (DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False)
               if world_size > 1 else model)
    if checkpoint is not None:
        rng = checkpoint["rng_by_rank"][rank]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        del checkpoint
    else:
        torch.manual_seed(config["seed"] + rank)
        torch.cuda.manual_seed(config["seed"] + rank)
    output_dir = Path(config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"kind": "start", "world_size": world_size, "queues": len(dataset),
                          "m1_samples": len(dataset.segmentation), "start_step": step,
                          "trainable_parameters": {group["name"]: sum(p.numel() for p in group["params"])
                                                   for group in optimizer.param_groups}}), flush=True)
    if world_size > 1:
        dist.barrier()
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    accumulation, max_steps = config["gradient_accumulation_steps"], config["max_steps"]
    wrapped.train()
    optimizer.zero_grad(set_to_none=True)
    while epoch < config["epochs"] and (max_steps is None or step < max_steps):
        sampler.set_epoch(epoch)
        dataset.epoch = epoch
        totals = torch.zeros(6, device=device, dtype=torch.float64)  # 两条loss/token数、队列数、前缀数。
        microbatches, started = 0, perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index < next_batch:
                started = perf_counter()
                continue
            batch = {branch: {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                              for key, value in content.items()} for branch, content in batch.items()}
            microbatches += 1
            epoch_finished = batch_index + 1 == len(loader)
            update = microbatches == accumulation or epoch_finished
            synchronization = wrapped.no_sync() if world_size > 1 and not update else nullcontext()
            with synchronization:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = wrapped(batch)
                    loss = result["m1_loss_sum"] + result["memory_loss_sum"]
                loss.backward()
            for index, key in enumerate(("m1_loss_sum", "m1_tokens", "memory_loss_sum", "memory_tokens")):
                totals[index] += result[key].detach().double()
            totals[4] += len(batch["memory"]["queue_event_counts"])
            totals[5] += len(batch["memory"]["state_descriptions"])
            if not update:
                continue
            if world_size > 1:
                dist.all_reduce(totals)
            normalize_gradients(optimizer, totals, world_size, config)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            next_epoch, cursor = (epoch + 1, 0) if epoch_finished else (epoch, batch_index + 1)
            stop = max_steps is not None and step >= max_steps
            if rank == 0 and (step % config["log_every_steps"] == 0 or epoch_finished or stop):
                torch.cuda.synchronize(device)
                seconds = perf_counter() - started
                m1_loss, text_loss = (totals[0] / totals[1]).item(), (totals[2] / totals[3]).item()
                record = {
                    "kind": "train", "epoch": epoch + 1, "step": step, "m1_loss": m1_loss, "text_loss": text_loss,
                    "loss": config["m1_loss_weight"] * m1_loss + config["memory_loss_weight"] * text_loss,
                    "m1_tokens": int(totals[1].item()), "text_tokens": int(totals[3].item()),
                    "queues_per_second": totals[4].item() / seconds, "prefixes_per_second": totals[5].item() / seconds,
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
            microbatches, started = 0, perf_counter()
            if stop:
                return
        epoch += 1
        next_batch = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", help="原始本地 Qwen3.5-0.8B 基座")
    parser.add_argument("--m1-checkpoint", help="第一次进入第二阶段时使用的 M1 LoRA 检查点")
    parser.add_argument("--m1-sample-files", nargs="+")
    parser.add_argument("--m2-sample-files", nargs="+")
    parser.add_argument("--data-manifest", help="批量数据清单，M1/M2均只读取train视频")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="第二阶段自己的 last.pt 或 epoch-XX.pt")
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None and checkpoint.get("format") != CHECKPOINT_FORMAT:
        parser.error("--resume 只接受第二阶段检查点；首次加载 M1 请使用 --m1-checkpoint")
    train(read_config(args, checkpoint), checkpoint)


if __name__ == "__main__":
    main()
