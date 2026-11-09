"""Single-GPU heads/full training with frozen splits, masked losses and resumable checkpoints."""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from simplememvla.data.reward import SimpleMemRewardCollator
from simplememvla.model import SimpleMemForRewardPrediction, SimpleMemRewardConfig
from simplememvla.training.data import RewardDataset


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--records", required=False)
    p.add_argument("--media-root")
    p.add_argument("--out")
    p.add_argument("--pairs")
    p.add_argument("--progress-only", action="store_true", help="Train progress without success or preference targets")
    p.add_argument("--mode", choices=["heads", "full"], default="heads")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--accumulate", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max-frames", type=int, default=240)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--views", choices=["main", "all"], default="all")
    p.add_argument("--allow-derived", action="store_true")
    p.add_argument("--eval-samples", type=int, default=0, help="0 evaluates the entire validation split")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume")
    known, _ = p.parse_known_args()
    if known.config:
        config = json.loads(Path(known.config).read_text())
        if set(config) - {a.dest for a in p._actions}:
            raise ValueError("Unknown config keys")
        p.set_defaults(**config)
    args = p.parse_args()
    if not args.records or not args.media_root or not args.out or min(args.epochs, args.accumulate, args.save_every) < 1:
        p.error("records, media-root, out and positive epochs/accumulate/save-every required")
    return args


def run(args):
    if args.progress_only and args.pairs:
        raise ValueError("Progress-only training does not use preference pairs")
    torch.manual_seed(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(Path(args.records).read_bytes()).hexdigest()
    if args.pairs:
        fingerprint += hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest()
    datasets = [
        RewardDataset(
            args.records,
            args.media_root,
            split,
            args.fps,
            args.max_frames,
            args.width,
            args.height,
            args.views,
            args.allow_derived,
            args.pairs,
        )
        for split in ("train", "val")
    ]
    processor = AutoProcessor.from_pretrained(args.model)
    processor.video_processor.do_resize = False  # Dataset already fixes each camera's pixel budget.
    processor.tokenizer.model_max_length = 262144
    cuda = str(args.device).startswith("cuda")
    # Full mode uses FP32 master weights; budget >= ~64 GiB before activations/temporary buffers.
    dtype = torch.bfloat16 if cuda and args.mode == "heads" else torch.float32
    model = SimpleMemForRewardPrediction.from_backbone(
        args.model,
        config=SimpleMemRewardConfig(freeze_backbone=args.mode == "heads"),
        dtype=dtype,
        attn_implementation="sdpa",
        device_map=args.device,
    )
    if args.pairs:
        model.prepare_processor(processor)
    for head in (model.progress_head, model.success_head, model.preference_head):
        head.float()
    # Zero masked losses still yield zero gradients; AdamW would decay those
    # weights. Freeze heads with no supervision rather than silently changing them.
    supervision = dict(
        progress=any(x["progress"] is not None for x in datasets[0].records),
        success=not args.progress_only and any(x["success"] is not None for x in datasets[0].records),
        preference=bool(args.pairs),
    )
    if args.progress_only and any(x["progress"] is None for data in datasets for x in data.records):
        raise ValueError("Progress-only training requires progress supervision for every selected record")
    for name, enabled in supervision.items():
        getattr(model, name + "_head").requires_grad_(enabled)
    print(
        json.dumps(dict(supervision=supervision, train_samples=len(datasets[0]), val_samples=len(datasets[1]))),
        flush=True,
    )
    if args.mode == "full":
        model.gradient_checkpointing_enable({"use_reentrant": False})
    collator = SimpleMemRewardCollator(processor)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    groups = [dict(params=head_params, lr=args.lr)]
    if args.mode == "full":
        groups.append(dict(params=[p for p in model.backbone.parameters() if p.requires_grad], lr=args.backbone_lr))
    # Full FP32 AdamW already retains large parameter/gradient/moment tensors;
    # foreach adds another parameter-sized temporary at step(), exceeding 80 GiB.
    optimizer = torch.optim.AdamW(groups, foreach=False if args.mode == "full" else None)
    step = start_epoch = start_position = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Resume dataset/pairs differ from checkpoint")
        for key in ("mode", "model", "fps", "max_frames", "width", "height", "views", "seed", "accumulate"):
            if saved["args"][key] != getattr(args, key):
                raise ValueError(f"Resume setting differs: {key}")
        if saved["args"].get("progress_only", False) != args.progress_only:
            raise ValueError("Resume objective differs: progress_only")
        model.load_state_dict(saved["model"], strict=args.mode == "full")
        if saved.get("preference_embedding") is not None:
            with torch.no_grad():
                model.backbone.get_input_embeddings().weight[model.config.preference_token_id].copy_(
                    saved["preference_embedding"]
                )
        optimizer.load_state_dict(saved["optimizer"])
        step, start_epoch, start_position = saved["step"], saved["epoch"], saved["position"]
        torch.set_rng_state(saved["rng"])
        if cuda:
            torch.cuda.set_rng_state(saved["cuda_rng"], device=args.device)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    processor.save_pretrained(output / "processor")

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext()

    def forward(feature):
        batch = collator([feature])
        if args.progress_only:
            batch.pop("success_targets", None)
            batch.pop("preference_targets", None)
        batch = {
            k: v.to(args.device, dtype=dtype if v.is_floating_point() and k.startswith("pixel") else v.dtype)
            for k, v in batch.items()
        }
        with autocast():
            result = model(**batch)
        return result, batch

    def save(epoch, position):
        state = model.state_dict()
        if args.mode == "heads":
            state = {k: v for k, v in state.items() if not k.startswith("backbone.")}
        temporary = output / "last.tmp"
        torch.save(
            dict(
                model=state,
                optimizer=optimizer.state_dict(),
                step=step,
                epoch=epoch,
                position=position,
                args=vars(args),
                fingerprint=fingerprint,
                rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(args.device) if cuda else None,
                preference_embedding=(
                    model.backbone.get_input_embeddings().weight[model.config.preference_token_id].detach().cpu()
                    if args.pairs and args.mode == "heads"
                    else None
                ),
            ),
            temporary,
        )
        temporary.replace(output / "last.pt")

    train, val = datasets
    for epoch in range(start_epoch, args.epochs):
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        position = start_position if epoch == start_epoch else 0
        while position < len(order) and (not args.max_steps or step < args.max_steps):
            model.train()
            if args.mode == "heads":
                model.backbone.eval()
            group = order[position : position + args.accumulate]
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for index in group:
                prediction, batch = forward(train[index])
                if prediction.loss is None or not torch.isfinite(prediction.loss):
                    raise ValueError("Missing/nonfinite training loss")
                (prediction.loss / len(group)).backward()
                loss_value += float(prediction.loss.detach()) / len(group)
                del prediction, batch
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            step += 1
            position += len(group)
            print(json.dumps(dict(step=step, epoch=epoch, position=position, loss=loss_value)), flush=True)
            if step % args.save_every == 0:
                save(epoch, position)
        save(epoch, position)
        if args.max_steps and step >= args.max_steps:
            break
    model.eval()
    errors, count, losses = 0.0, 0, []
    limit = min(len(val), args.eval_samples) if args.eval_samples else len(val)
    with torch.inference_mode():
        for index in range(limit):
            result, batch = forward(val[index])
            valid = batch["progress_targets"] != -100
            errors += float((result.progress[valid] - batch["progress_targets"][valid]).abs().sum())
            count += int(valid.sum())
            losses.append(float(result.loss))
    metrics = dict(
        step=step,
        validation_samples=limit,
        progress_mae=errors / count if count else None,
        validation_loss=sum(losses) / len(losses),
        supervised_progress_patches=count,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)
    return metrics


if __name__ == "__main__":
    run(arguments())
