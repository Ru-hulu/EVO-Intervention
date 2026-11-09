"""Train Qwen3.5 VST memory with chronological windows and truncated BPTT."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from mem_lite.datasets.vst_dataset import EpisodeWindows, WindowCollator
from mem_lite.models.vst_predictor import VSTProgressPredictor

CHECKPOINT_FORMAT = "mem-lite-qwen35-vst-qkv-shared-o-v2"


def trainable_state(model):
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_trainable_state(model, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("VST checkpoint trainable parameter names disagree")
    model.load_state_dict(state, strict=False)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(model, optimizer, config, output, epoch, next_window, step, epoch_stats):
    # Only at episode-group boundaries: no live video state needs serialization.
    rng = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
    states = [None] * dist.get_world_size() if dist.is_initialized() else [rng]
    if dist.is_initialized():
        dist.all_gather_object(states, rng)
    if not dist.is_initialized() or dist.get_rank() == 0:
        temporary = output / "last.pt.tmp"
        torch.save({"format": CHECKPOINT_FORMAT, "model": trainable_state(model), "optimizer": optimizer.state_dict(),
                    "config": config, "epoch": epoch, "next_window": next_window, "step": step,
                    "rng_by_rank": states, "epoch_stats": epoch_stats.detach().cpu()}, temporary)
        temporary.replace(output / "last.pt")
    if dist.is_initialized():
        dist.barrier()


def device_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def load_config(path):
    config = json.loads(Path(path).read_text())
    if isinstance(config["sample_files"], str):
        config["sample_files"] = json.loads(Path(config["sample_files"]).read_text())
    for key in ("epochs", "batch_size", "window_size", "compression_ratio", "learning_rate"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return config


def train(config, resume=None):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        _train(config, resume, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train(config, resume, rank, world, device):
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("VST training requires BF16 CUDA hardware")
    # Arbitrary block masks must not silently fall back to quadratic math SDPA.
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    processor = AutoProcessor.from_pretrained(config["model_path"], max_pixels=config["max_pixels"], local_files_only=True)
    model = VSTProgressPredictor(config["model_path"], **config["model"]).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"],
                                  fused=config.get("fused_optimizer", True))
    epoch, cursor, step = 0, 0, 0
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    resume_stats = None
    if state is not None:
        if state["format"] != CHECKPOINT_FORMAT or len(state["rng_by_rank"]) != world:
            raise ValueError("Checkpoint format/world size mismatch")
        for key in ("batch_size", "window_size", "compression_ratio", "seed", "model", "sample_files"):
            if state["config"][key] != config[key]:
                raise ValueError(f"Resume requires identical {key}")
        load_trainable_state(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, cursor, step = state["epoch"], state["next_window"], state["step"]
        resume_stats = state["epoch_stats"].to(device)
    wrapped = DistributedDataParallel(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    if state is not None:
        torch.set_rng_state(state["rng_by_rank"][rank]["torch"])
        torch.cuda.set_rng_state(state["rng_by_rank"][rank]["cuda"], device)
        expected_signature = state["config"]["data_signature"]
        del state
    else:
        torch.manual_seed(config["seed"] + rank)
        torch.cuda.manual_seed_all(config["seed"] + rank)
        expected_signature = None
    output = Path(config["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if not resume and (output / "train.jsonl").exists():
            raise ValueError("Existing training output: choose a fresh directory or --resume")
        print(json.dumps({"kind": "start", "world_size": world,
                          "trainable_parameters": sum(p.numel() for p in parameters), "config": config}), flush=True)
    if world > 1:
        dist.barrier()
    wrapped.train()
    last_save = step
    while epoch < config["epochs"]:
        dataset = EpisodeWindows(config["sample_files"], config["batch_size"], config["window_size"],
                                 rank, world, config["seed"], epoch)
        if expected_signature is not None and dataset.signature != expected_signature:
            raise ValueError("Dataset changed since checkpoint")
        config["data_signature"] = dataset.signature
        if cursor and not dataset.windows[cursor]["reset"]:
            raise ValueError("Resume cursor must be an episode-group boundary")
        loader = DataLoader(Subset(dataset, range(cursor, len(dataset))), batch_size=None,
            collate_fn=WindowCollator(processor, config["compression_ratio"], config["max_sequence_length"]),
            num_workers=config["num_workers"], pin_memory=True,
            multiprocessing_context="spawn" if config["num_workers"] else None,
            persistent_workers=config["num_workers"] > 0,
            generator=torch.Generator().manual_seed(config["seed"] + rank + epoch * world),
        )
        if rank == 0:
            write_json(output / "config.json", config)
        cache, epoch_stats = None, torch.zeros(5, dtype=torch.float64, device=device)
        if resume_stats is not None:
            epoch_stats.copy_(resume_stats)
            resume_stats = None
        began_epoch = perf_counter()
        ready = perf_counter()
        for index, item in enumerate(loader, start=cursor):
            loaded = perf_counter()
            inputs = device_batch(item["inputs"], device)
            target = item["target_pt"].to(device, non_blocking=True)
            if item["reset"]:
                cache = None
            optimizer.zero_grad(set_to_none=True)
            result = wrapped(inputs, target, cache)
            result["loss_sum"].backward()
            active = inputs["frame_valid"]
            error = (result["pt"].detach() - target).double()[active]
            stats = torch.stack((result["loss_sum"].detach().double(), active.sum().double(),
                                 error.abs().sum(), error.square().sum(), error.sum()))
            if world > 1:
                dist.all_reduce(stats)
            if stats[1] <= 0:
                raise RuntimeError("Globally empty training window")
            scale = world / stats[1]
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            # All past KV remain available; only the autograd graph is truncated.
            cache = result["cache"].detach()
            epoch_stats += stats
            step += 1
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - ready
            epoch_finished = index + 1 == len(dataset)
            stop = config.get("max_steps") is not None and step >= config["max_steps"]
            if step == 1 or step % config["log_every_steps"] == 0 or epoch_finished or stop:
                memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                                       torch.cuda.max_memory_reserved(device) / 2**30,
                                       cache.lengths.max().item(), elapsed, loaded - ready], device=device)
                all_memory = [torch.empty_like(memory) for _ in range(world)]
                if world > 1:
                    dist.all_gather(all_memory, memory)
                else:
                    all_memory = [memory]
                if rank == 0:
                    seconds = max(float(m[3]) for m in all_memory)
                    record = {"kind": "train", "epoch": epoch + 1, "step": step, "window": index + 1,
                        "windows_per_epoch": len(dataset), "samples": int(stats[1]), "loss": float(stats[0] / stats[1]),
                        "pt_mae_pp": float(100 * stats[2] / stats[1]), "grad_norm": float(norm),
                        "step_seconds": seconds, "samples_per_second": float(stats[1]) / seconds,
                        "data_seconds_per_gpu": [float(m[4]) for m in all_memory],
                        "peak_allocated_gib_per_gpu": [float(m[0]) for m in all_memory],
                        "peak_reserved_gib_per_gpu": [float(m[1]) for m in all_memory],
                        "history_tokens_per_gpu": [int(m[2]) for m in all_memory]}
                    print(json.dumps(record), flush=True)
                    with (output / "train.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
            save_due = item["group_end"] and (epoch_finished or step - last_save >= config["save_every_steps"])
            if save_due:
                next_epoch, next_window = (epoch + 1, 0) if epoch_finished else (epoch, index + 1)
                save_checkpoint(model, optimizer, config, output, next_epoch, next_window, step,
                                torch.zeros_like(epoch_stats) if epoch_finished else epoch_stats)
                last_save = step
            del result
            if stop:
                # Benchmarks deliberately stop mid-video without a resumable checkpoint.
                if rank == 0:
                    write_json(output / "benchmark-finished.json", {"step": step, "epochs_completed": epoch,
                        "resumable_checkpoint": (output / "last.pt").exists()})
                return
            ready = perf_counter()
        if rank == 0:
            report = {"epoch": epoch + 1, "samples": int(epoch_stats[1]),
                      "loss": float(epoch_stats[0] / epoch_stats[1]), "mae_pp": float(100 * epoch_stats[2] / epoch_stats[1]),
                      "rmse_pp": float(100 * (epoch_stats[3] / epoch_stats[1]).sqrt()),
                      "seconds": perf_counter() - began_epoch, "steps": step}
            if report["samples"] != len(dataset.frames):
                raise RuntimeError("Epoch did not cover every observation exactly once")
            with (output / "epochs.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")
        epoch, cursor = epoch + 1, 0
    if rank == 0:
        write_json(output / "training-completed.json", {"epochs": epoch, "steps": step, "checkpoint": str(output / "last.pt")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
