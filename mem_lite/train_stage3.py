"""三视角 + StreamForest 缓存，只训练语言 LoRA、历史投影和进度头。"""

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

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.memory.vision import verify_vision_source
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
    for key in ("model_path", "sample_files", "output_dir", "epochs", "max_steps"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if not config["model_path"] or not config["sample_files"]:
        raise ValueError("需要本地 Qwen3.5 基座及已构造的 StreamForest 样本")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
                "log_every_steps", "save_every_steps", "max_grad_norm"):
        if config[key] <= 0:
            raise ValueError(f"{key} 必须大于0")
    if config["max_steps"] is not None and config["max_steps"] < 1:
        raise ValueError("max_steps 必须为正整数或 null")
    for key in ("model_path", "output_dir"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    config["sample_files"] = [str(Path(path).expanduser().resolve()) for path in config["sample_files"]]
    return config


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
    config["model"]["history_dim"] = dataset.history_dim
    config["history_capacity"] = dataset.history_capacity
    config["history_source"] = dataset.history_source


def train(config, checkpoint=None):
    if not torch.cuda.is_available():
        raise RuntimeError("第三阶段训练需要 CUDA；本地仅开发及离线检查，不应调用 train")
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
        raise RuntimeError("第三阶段训练需要支持 BF16 的 GPU")
    if checkpoint is not None and len(checkpoint["rng_by_rank"]) != world_size:
        raise ValueError("断点续训需保持 GPU 数量，以恢复相同的样本顺序及梯度累积")
    torch.manual_seed(config["seed"])
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
        num_workers=config["num_workers"], pin_memory=True, persistent_workers=False,
        multiprocessing_context="spawn" if config["num_workers"] > 0 else None,
        generator=torch.Generator().manual_seed(config["seed"] + rank),
    )
    model = M3ProgressPredictor(config["model_path"], **config["model"]).to(device)
    if dataset.history_dim != model.qwen.config.vision_config.out_hidden_size:
        raise ValueError("缓存的视觉特征维度与 Qwen 基座不一致")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epoch, next_batch, step = 0, 0, 0
    if checkpoint is not None:
        load_trainable_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch, next_batch, step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"]
    # 视频首帧没有历史，某个 rank/微批次可能不使用 history_projection。
    wrapped = (DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False,
                                       find_unused_parameters=True) if world_size > 1 else model)
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
            optimizer.zero_grad(set_to_none=True)
            step += 1
            next_epoch, cursor = (epoch + 1, 0) if epoch_finished else (epoch, batch_index + 1)
            stop = max_steps is not None and step >= max_steps
            if rank == 0 and (step % config["log_every_steps"] == 0 or epoch_finished or stop):
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
    parser.add_argument("--model-path", help="原始本地 Qwen3.5 基座")
    parser.add_argument("--sample-files", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", help="第三阶段自己的 last.pt 或 epoch-XX.pt")
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint is not None and checkpoint.get("format") != CHECKPOINT_FORMAT:
        parser.error("--resume 只接受 StreamForest 进度模型检查点")
    train(read_config(args, checkpoint), checkpoint)


if __name__ == "__main__":
    main()
