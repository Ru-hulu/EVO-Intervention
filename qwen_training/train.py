# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License; see LICENSE.
# Original trainer implemented by Jinhui YE / HKUST University (2025).
# ruff: noqa: RUF001, RUF003
"""真实视频样本 → MEM 视觉特征 → JSON 答案损失 → 更新 LoRA 和视觉适配层。"""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from qwen_training.dataloader import build_dataloaders
from qwen_training.dataset import augment_training_images
from qwen_training.model import QwenProgressModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="本地 Qwen3.5-4B 权重目录")
    parser.add_argument("--index-files", nargs="+", help="一个或多个 samples.jsonl 路径")
    parser.add_argument("--output-dir", default="outputs/qwen_training")
    parser.add_argument("--resume", help="从 last.pt 恢复训练")
    parser.add_argument("--steps", type=int, default=20, help="目标总步数，包含已完成的步数")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-pixels", type=int, default=262144, help="每张图像预处理后的最大像素数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "lora" not in checkpoint:
            parser.error("此入口恢复 LoRA 检查点；旧适配层检查点请重新开始训练")
        # 恢复模型与数据配置；总步数、设备、输出目录和读取进程数可在本次指定。
        for key in ("model_path", "index_files", "batch_size", "val_fraction", "max_pixels", "lr", "seed",
                    "lora_rank", "lora_alpha", "lora_dropout"):
            setattr(args, key, checkpoint["config"][key])
    if not args.model_path or not args.index_files:
        parser.error("首次训练需要 --model-path 和 --index-files；恢复训练使用 --resume")
    if args.lora_rank < 1:
        parser.error("LoRA 训练要求 --lora-rank >= 1")

    torch.manual_seed(args.seed)
    train_loader, val_loader = build_dataloaders(
        args.index_files, batch_size=args.batch_size, num_workers=args.num_workers,
        val_fraction=args.val_fraction, seed=args.seed,
    )
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = QwenProgressModel(
        args.model_path, dtype=dtype, freeze_backbone=True, max_pixels=args.max_pixels,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    ).to(device)
    # LoRA、视觉适配层和 AdamW 状态保留 FP32；CUDA 前向使用 BF16。
    parameters = [p for p in model.parameters() if p.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    model.train()
    model.qwen.model.visual.eval()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)

    start_step = 0
    if checkpoint is not None:
        model.visual_adapter.load_state_dict(checkpoint["visual_adapter"])
        set_peft_model_state_dict(model.qwen, checkpoint["lora"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        del checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} trainable={sum(p.numel() for p in parameters):,} "
          f"train_samples={len(train_loader.dataset)} val_samples={len(val_loader.dataset)} "
          f"steps={start_step}->{args.steps}", flush=True)

    step = start_step
    epoch, completed_batches = divmod(start_step, len(train_loader))
    while step < args.steps:
        # 比例配额固定；每轮由 seed + epoch 决定类内抽样和整体顺序，恢复时重建。
        train_loader.sampler.generator.manual_seed(args.seed + epoch)
        batch_started = perf_counter()
        for batch_index, (inputs, targets) in enumerate(train_loader):
            if batch_index < completed_batches:
                batch_started = perf_counter()
                continue
            # 仅训练执行；18 张图共享轻量增强，恢复到同一步时参数相同。
            inputs["images"] = augment_training_images(
                inputs["images"], inputs["messages"], targets["responses"], seed=args.seed + step,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                # 视觉编码器冻结；语言前向保留计算图，更新 LoRA 和视觉适配层。
                with torch.no_grad():
                    visual = model.encode_history(inputs["images"], frame_mask=inputs["frame_mask"])
                loss = model(messages=inputs["messages"], **visual, **targets).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_seconds = perf_counter() - batch_started

            if step % args.log_every == 0 or step == args.steps:
                print(f"step={step}/{args.steps} epoch={epoch + 1} "
                      f"batch={batch_index + 1}/{len(train_loader)} loss={loss.item():.6f} "
                      f"step_s={step_seconds:.3f}", flush=True)

            if step % args.save_every == 0 or step == args.steps:
                # step 同时记录数据读取位置；每轮批数由相同索引与 batch_size 确定。
                temporary = output_dir / "last.pt.tmp"
                torch.save({
                    "visual_adapter": model.visual_adapter.state_dict(),
                    "lora": get_peft_model_state_dict(model.qwen, save_embedding_layers=False),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": vars(args),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                }, temporary)
                temporary.replace(output_dir / "last.pt")
                print(f"saved={output_dir / 'last.pt'}", flush=True)
            if step == args.steps:
                break
            batch_started = perf_counter()
        epoch += 1
        completed_batches = 0


if __name__ == "__main__":
    main()
