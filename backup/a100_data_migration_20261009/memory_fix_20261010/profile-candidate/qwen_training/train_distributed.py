"""多卡按 epoch 训练：全局类别配额、独立验证集、终端日志、指定轮次保存。"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
from time import perf_counter, time

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from qwen_training.dataloader import build_dataloaders
from qwen_training.dataset import augment_training_images, collate_fn
from qwen_training.model import QwenProgressModel
from qwen_training.prompts import build_messages


class GlobalBatchSampler(Sampler):
    """全局类别配额只生成一次；末批过小时从前一批调剂，不丢弃或重复样本。"""

    def __init__(self, sampler, batch_size, rank, world_size, seed):
        self.sampler, self.batch_size = sampler, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.epoch = 0
        if not 0 <= rank < world_size or batch_size < world_size:
            raise ValueError("rank 必须有效，global-batch-size 必须不少于 GPU 数")
        count = len(sampler)
        if count < world_size:
            raise ValueError("训练样本数必须不少于 GPU 数")
        self.batch_sizes = [batch_size] * (count // batch_size)
        if count % batch_size:
            self.batch_sizes.append(count % batch_size)
        if self.batch_sizes[-1] < world_size:
            borrowed = world_size - self.batch_sizes[-1]
            if len(self.batch_sizes) < 2 or self.batch_sizes[-2] - borrowed < world_size:
                raise ValueError("末批无法分给所有 GPU；请使用 GPU 数的至少两倍作为 global-batch-size")
            self.batch_sizes[-2] -= borrowed
            self.batch_sizes[-1] += borrowed

    def __iter__(self):
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        start = 0
        for size in self.batch_sizes:
            batch = indices[start:start + size]
            start += size
            yield batch[self.rank::self.world_size]

    def __len__(self):
        return len(self.batch_sizes)


def add_metrics(totals, outputs, targets):
    # 累计 CE 总和、token 数、平方误差总和、绝对误差总和、样本数。
    count = len(targets)
    totals += torch.stack((
        outputs.text_loss.detach().double() * outputs.text_tokens,
        outputs.text_tokens.double(),
        outputs.progress_loss.detach().double() * count,
        (outputs.progress.detach() - targets.to(outputs.progress)).abs().double().sum(),
        totals.new_tensor(count),
    ))


def collect_metrics(totals, args):
    totals = totals.clone()
    dist.all_reduce(totals)
    text_sum, token_count, squared_error, absolute_error, count = totals.tolist()
    text_loss, progress_loss = text_sum / token_count, squared_error / count
    return {
        "loss": args.text_loss_weight * text_loss + args.progress_loss_weight * progress_loss,
        "text_ce": text_loss, "progress_mse": progress_loss,
        "progress_mae": absolute_error / count, "samples": int(count),
    }


def phase_batches(loader):
    """耗尽本阶段数据后等待它创建的 worker 退出，不触碰框架的辅助进程。"""
    existing = {child.pid for child in multiprocessing.active_children()}
    iterator = iter(loader)
    workers = [child for child in multiprocessing.active_children() if child.pid not in existing]
    try:
        yield from iterator
    finally:
        del iterator
        for worker in workers:
            worker.join(timeout=30)
        alive = [worker.pid for worker in workers if worker.is_alive()]
        if alive:
            raise RuntimeError(f"本阶段 DataLoader 进程未退出：{alive}")


@torch.no_grad()
def evaluate(model, loader, args, rank, device):
    model.eval()
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    for batch_index, (inputs, targets) in enumerate(phase_batches(loader)):
        if args.gpu_image_processing:
            inputs["images"] = inputs["images"].to(device, non_blocking=True)
        # 每轮使用同一批验证视频、同一提示词；不做图像增强或类别重采样。
        messages = build_messages(
            inputs["input_states"], state_drop_probability=0.3,
            seed=args.seed + 1_000_000 + batch_index * dist.get_world_size() + rank,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            visual = model.encode_history(inputs["images"], inputs["frame_mask"])
            outputs = model(messages=messages, **visual, **targets)
        add_metrics(totals, outputs, targets["progress"])
    result = collect_metrics(totals, args)
    model.train()
    model.qwen.model.visual.eval()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--index-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--save-epochs", type=int, nargs="+", default=[3, 6, 8])
    parser.add_argument("--global-batch-size", type=int, default=10)
    parser.add_argument("--val-batch-size", type=int, default=5, help="每卡验证 batch，无梯度")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="每个数据读取进程提前准备的 batch 数，降低视频解码时长波动")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="语言层使用非重入梯度检查点，以重算换取更低激活显存")
    parser.add_argument("--require-fast-kernels", action="store_true",
                        help="缺少 Qwen3.5 的 FLA/causal-conv1d 加速算子时提前报错")
    parser.add_argument("--gpu-image-processing", action="store_true",
                        help="将 uint8 帧整体传入本卡，在 GPU 上进行原有图像增强和预处理")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text-loss-weight", type=float, default=0.7)
    parser.add_argument("--progress-loss-weight", type=float, default=0.3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", help="恢复本入口在 epoch 末尾保存的检查点")
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint:
        for key, value in checkpoint["config"].items():
            if key not in {"epochs", "save_epochs", "output_dir", "resume", "num_workers", "prefetch_factor", "log_every"}:
                setattr(args, key, value)

    if args.require_fast_kernels:
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
        if not is_fast_path_available:
            parser.error("请先安装兼容的 flash-linear-attention 和 causal-conv1d")

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed)

    # 复用现有视频划分与全局类别比例，只替换各卡的 batch 分配方式。
    base_train, base_val = build_dataloaders(
        args.index_files, num_workers=0, val_fraction=args.val_fraction, seed=args.seed,
    )
    sampler = GlobalBatchSampler(base_train.sampler, args.global_batch_size, rank, world_size, args.seed)
    options = dict(
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
        multiprocessing_context="spawn" if args.num_workers else None,
        # 每阶段耗尽 loader 后退出 worker，避免训练和验证两套解码器/索引副本同时驻留。
        persistent_workers=False,
        prefetch_factor=args.prefetch_factor if args.num_workers else None,
    )
    train_loader = DataLoader(
        base_train.dataset, batch_sampler=sampler,
        generator=torch.Generator().manual_seed(args.seed + rank + 1), **options,
    )
    val_loader = DataLoader(
        base_val.dataset, sampler=range(rank, len(base_val.dataset), world_size),
        batch_size=args.val_batch_size,
        generator=torch.Generator().manual_seed(args.seed + rank + 100), **options,
    )
    model = QwenProgressModel(
        args.model_path, dtype=torch.bfloat16, freeze_backbone=True, max_pixels=args.max_pixels,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        text_loss_weight=args.text_loss_weight, progress_loss_weight=args.progress_loss_weight,
    ).to(device)
    if args.gradient_checkpointing:
        model.qwen.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    parameters = [p for p in model.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    if checkpoint:
        model.visual_adapter.load_state_dict(checkpoint["visual_adapter"])
        model.progress_head.load_state_dict(checkpoint["progress_head"])
        set_peft_model_state_dict(model.qwen, checkpoint["lora"])
    distributed_model = DistributedDataParallel(
        model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    step, start_epoch = 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, start_epoch = checkpoint["step"], checkpoint["epoch"]
        rng = checkpoint["rng_by_rank"][rank]
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        del checkpoint
    else:
        # 模型初始化相同，训练时不同卡的 dropout 使用不同随机流。
        torch.manual_seed(args.seed + rank)
    model.train()
    model.qwen.model.visual.eval()

    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
        split = {name: sorted({(dataset.dataset.samples[i]["dataset"], dataset.dataset.samples[i]["video_id"])
                              for i in dataset.indices})
                 for name, dataset in (("train", base_train.dataset), ("validation", base_val.dataset))}
        (output / "video_split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2))
        print(f"START world_size={world_size} global_batch={args.global_batch_size} "
              f"train={len(base_train.dataset)} val={len(base_val.dataset)} "
              f"updates_per_epoch={len(train_loader)} epochs={start_epoch}->{args.epochs}", flush=True)

    def record(kind, metrics, epoch):
        if rank != 0:
            return
        row = {"kind": kind, "step": step, "epoch": epoch, "timestamp": time(), **metrics}
        print(json.dumps(row), flush=True)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def record_phase(phase, epoch):
        # phase_batches 已等待本阶段 worker 退出；同步所有 rank 后才能创建下一批 worker。
        dist.barrier()
        if rank == 0:
            memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            record("phase", {
                "phase": phase, "active_data_workers": 0,
                "host_available_gib": int(memory["MemAvailable"].split()[0]) / 2**20,
            }, epoch)

    for epoch in range(start_epoch, args.epochs):
        record_phase("before_train", epoch + 1)
        sampler.epoch = epoch
        epoch_totals = torch.zeros(5, dtype=torch.float64, device=device)
        window_totals = torch.zeros_like(epoch_totals)
        epoch_started = window_started = perf_counter()
        _profile_loop_end = perf_counter()
        for batch_index, (inputs, targets) in enumerate(phase_batches(train_loader)):
            _profile_started = perf_counter()
            _data_wait = _profile_started - _profile_loop_end
            _events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
            _events[0].record()
            seed = args.seed + step * world_size + rank
            messages = build_messages(inputs["input_states"], state_drop_probability=0.3, seed=seed)
            if args.gpu_image_processing:
                inputs["images"] = inputs["images"].to(device, non_blocking=True)
            inputs["images"] = augment_training_images(inputs["images"], inputs["state_pairs"], seed)
            optimizer.zero_grad(set_to_none=True)
            _events[1].record()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    visual = model.encode_history(inputs["images"], inputs["frame_mask"])
                _events[2].record()
                outputs = distributed_model(messages=messages, **visual, **targets)
                _events[3].record()
                count = len(targets["progress"])
                denominators = torch.stack((outputs.text_tokens.double(), window_totals.new_tensor(count)))
                dist.all_reduce(denominators)
                # DDP 对卡取平均，乘 world_size 抵消；CE 按 token，MSE 按样本加权。
                loss = world_size * (
                    args.text_loss_weight * outputs.text_loss * outputs.text_tokens / denominators[0]
                    + args.progress_loss_weight * outputs.progress_loss * count / denominators[1]
                )
            _events[4].record()
            loss.backward()
            _events[5].record()
            add_metrics(window_totals, outputs, targets["progress"])
            add_metrics(epoch_totals, outputs, targets["progress"])
            del outputs, visual, loss
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            _events[6].record()
            torch.cuda.synchronize()
            _stage_names = ["prompt_transfer_augmentation", "vision", "language_forward", "loss_and_denominator_sync", "backward_with_ddp", "metrics_clip_optimizer"]
            _profile_row = {"rank": rank, "step": step + 1, "data_wait_seconds": _data_wait, "compute_wall_seconds": perf_counter() - _profile_started, "gpu_stage_ms": {name: _events[i].elapsed_time(_events[i+1]) for i, name in enumerate(_stage_names)}}
            with (output / f"profile-rank-{rank}.jsonl").open("a") as _profile_file:
                _profile_file.write(json.dumps(_profile_row) + "\n")
            step += 1
            if step % args.log_every == 0 or batch_index + 1 == len(train_loader):
                metrics = collect_metrics(window_totals, args)
                elapsed = perf_counter() - window_started
                metrics["samples_per_second"] = metrics["samples"] / elapsed
                record("train", metrics, epoch + 1)
                window_totals.zero_()
                window_started = perf_counter()
            _profile_loop_end = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        metrics = collect_metrics(epoch_totals, args)
        metrics["seconds"] = perf_counter() - epoch_started
        record("train_epoch", metrics, epoch + 1)
        record_phase("before_validation", epoch + 1)
        validation_started = perf_counter()
        metrics = evaluate(model, val_loader, args, rank, device)
        metrics["seconds"] = perf_counter() - validation_started
        record("validation", metrics, epoch + 1)
        record_phase("after_validation", epoch + 1)
        if epoch + 1 in args.save_epochs:
            rng_by_rank = [None] * world_size
            dist.all_gather_object(rng_by_rank, {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)})
            if rank == 0:
                path = output / f"epoch-{epoch + 1:02d}.pt"
                temporary = path.with_suffix(".pt.tmp")
                torch.save({
                    "visual_adapter": model.visual_adapter.state_dict(),
                    "progress_head": model.progress_head.state_dict(),
                    "lora": get_peft_model_state_dict(model.qwen, save_embedding_layers=False),
                    "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch + 1,
                    "config": vars(args), "world_size": world_size, "rng_by_rank": rng_by_rank,
                }, temporary)
                temporary.replace(path)
                print(f"SAVED {path}", flush=True)
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
