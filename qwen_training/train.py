# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License; see LICENSE.
# Original trainer implemented by Jinhui YE / HKUST University (2025).
"""最小训练入口：fakefunc → JSON 答案损失 → 更新视觉适配层。"""

import argparse
from pathlib import Path

import torch

from qwen_training.fake_data import fakefunc
from qwen_training.model import QwenProgressModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="本地 Qwen3.5-4B 权重目录")
    parser.add_argument("--output-dir", default="outputs/qwen_training")
    parser.add_argument("--resume", help="从 last.pt 恢复训练")
    parser.add_argument("--steps", type=int, default=20, help="目标总步数，包含已完成的步数")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        # 恢复相同的模型配置和假数据；总步数、设备和输出目录仍由本次命令指定。
        for key in ("model_path", "feature_dim", "num_tokens", "batch_size", "lr", "seed"):
            setattr(args, key, checkpoint["config"][key])
    if not args.model_path:
        parser.error("首次训练需要 --model-path；恢复训练使用 --resume")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = QwenProgressModel(
        args.model_path, feature_dim=args.feature_dim, dtype=dtype, freeze_backbone=True
    ).to(device)
    # 适配层参数和 AdamW 状态保留 FP32；CUDA 前向使用 BF16 混合精度。
    model.visual_adapter.float()
    model.train()
    model.qwen.eval()
    parameters = list(model.visual_adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)

    start_step = 0
    if checkpoint is not None:
        model.visual_adapter.load_state_dict(checkpoint["visual_adapter"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        del checkpoint

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} trainable={sum(p.numel() for p in parameters):,} "
          f"steps={start_step}->{args.steps}", flush=True)

    for step in range(start_step + 1, args.steps + 1):
        # 固定的三个假样本仅用于跑通链路；后续在这里替换为真实数据。
        inputs, targets = fakefunc(
            batch_size=args.batch_size, num_tokens=args.num_tokens,
            feature_dim=args.feature_dim, seed=args.seed, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss = model(**inputs, **targets).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps:
            print(f"step={step}/{args.steps} loss={loss.item():.6f}", flush=True)

        if step % args.save_every == 0 or step == args.steps:
            # 主干冻结，恢复时从 model_path 重载，只保存实际训练的参数。
            temporary = output_dir / "last.pt.tmp"
            torch.save({
                "visual_adapter": model.visual_adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": vars(args),
            }, temporary)
            temporary.replace(output_dir / "last.pt")
            print(f"saved={output_dir / 'last.pt'}", flush=True)


if __name__ == "__main__":
    main()
