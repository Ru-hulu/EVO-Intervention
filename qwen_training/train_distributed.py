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
from qwen_training.dataset import augment_training_images, collate_fn, prepare_training_batch
from qwen_training.model import QwenProgressModel
from qwen_training.prompts import INPUT_COMBINATIONS, MISMATCHED_PROMPTS


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


def add_metrics(totals, outputs, targets, combination_ids):
    # 统一按所有有效答案 token 累计 CE；数值误差由自回归生成评估单独计算。
    totals[:3] += torch.stack((
        outputs.text_loss.detach().double() * outputs.text_tokens,
        outputs.text_tokens.double(),
        totals.new_tensor(len(targets["responses"])),
    ))
    totals[3:] += torch.bincount(torch.tensor(combination_ids, device=totals.device),
                                minlength=len(INPUT_COMBINATIONS))


def collect_metrics(totals):
    totals = totals.clone()
    dist.all_reduce(totals)
    text_sum, token_count, count = totals[:3].tolist()
    text_loss = text_sum / max(token_count, 1)
    return {
        "loss": text_loss, "text_ce": text_loss, "text_tokens": int(token_count), "samples": int(count),
        "input_combination_counts": {
            f"{mode}_{'memory' if keep else 'no_memory'}": int(n)
            for (mode, keep, _), n in zip(INPUT_COMBINATIONS, totals[3:].tolist(), strict=True)
        },
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
    totals = torch.zeros(3 + len(INPUT_COMBINATIONS), dtype=torch.float64, device=device)
    for batch_index, (inputs, targets) in enumerate(phase_batches(loader)):
        if args.gpu_image_processing:
            inputs["images"] = inputs["images"].to(device, non_blocking=True)
        # 每轮使用同一批验证视频、同一提示词；不做图像增强或类别重采样。
        targets = prepare_training_batch(
            inputs, mismatched_prompts=args.mismatched_prompts,
            seed=args.seed + 1_000_000 + batch_index * dist.get_world_size() + rank,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            visual = model.encode_history(inputs["images"], inputs["frame_mask"])
            outputs = model(messages=inputs["messages"], **visual, **targets)
        add_metrics(totals, outputs, targets, inputs["combination_ids"])
    result = collect_metrics(totals)
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
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mismatched-prompts", nargs="+", default=list(MISMATCHED_PROMPTS),
                        help="与实际装配任务语义不同的负任务提示词")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", help="恢复本入口在 epoch 末尾保存的检查点")
    args = parser.parse_args()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint:
        if "progress_head" in checkpoint:
            parser.error("旧回归头检查点不能直接续训，请使用文本进度检查点或重新训练")
        for key, value in checkpoint["config"].items():
            if key not in {"epochs", "save_epochs", "output_dir", "resume", "num_workers", "prefetch_factor", "log_every"}:
                setattr(args, key, value)
    args.input_combinations = INPUT_COMBINATIONS
    args.progress_format = "text_3dp"

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
        epoch_totals = torch.zeros(3 + len(INPUT_COMBINATIONS), dtype=torch.float64, device=device)
        window_totals = torch.zeros_like(epoch_totals)
        epoch_started = window_started = perf_counter()
        for batch_index, (inputs, targets) in enumerate(phase_batches(train_loader)):
            seed = args.seed + step * world_size + rank
            targets = prepare_training_batch(inputs, seed, mismatched_prompts=args.mismatched_prompts)
            if args.gpu_image_processing:
                inputs["images"] = inputs["images"].to(device, non_blocking=True)
            inputs["images"] = augment_training_images(inputs["images"], inputs["state_pairs"], seed)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    visual = model.encode_history(inputs["images"], inputs["frame_mask"])
                outputs = distributed_model(messages=inputs["messages"], **visual, **targets)
                token_count = outputs.text_tokens.double()
                dist.all_reduce(token_count)
                # DDP 对卡取平均；恢复为全局所有答案 token 的平均 CE。
                loss = world_size * outputs.loss * outputs.text_tokens / token_count.clamp_min(1)
            loss.backward()
            add_metrics(window_totals, outputs, targets, inputs["combination_ids"])
            add_metrics(epoch_totals, outputs, targets, inputs["combination_ids"])
            del outputs, visual, loss
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % args.log_every == 0 or batch_index + 1 == len(train_loader):
                metrics = collect_metrics(window_totals)
                elapsed = perf_counter() - window_started
                metrics["samples_per_second"] = metrics["samples"] / elapsed
                record("train", metrics, epoch + 1)
                window_totals.zero_()
                window_started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        metrics = collect_metrics(epoch_totals)
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
