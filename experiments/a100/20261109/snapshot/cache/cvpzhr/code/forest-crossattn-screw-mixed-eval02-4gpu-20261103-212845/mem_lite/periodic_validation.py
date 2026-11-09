"""Fractional-epoch validation without padding samples or changing training RNG state."""

from fractions import Fraction
import json
import math
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.models.m3_predictor import adjacent_bin_targets


def validation_points(updates_per_epoch, interval):
    fraction = Fraction(str(interval))
    if not 0 < fraction <= 1:
        raise ValueError("validation_interval_epochs must be in (0, 1]")
    fractions = [min(i * fraction, 1) for i in range(1, math.ceil(1 / fraction) + 1)]
    points = {math.ceil(updates_per_epoch * f): float(f) for f in fractions}
    if len(points) != len(fractions):
        raise ValueError("Too few optimizer steps for the requested validation interval")
    return points


def group_name(summary):
    source = summary.get("source", {})
    task = source.get("task", "all")
    rounds = source.get("source_episodes")
    return f"{task}/n{len(rounds)}" if rounds else task


def make_validation_loader(config, processor, training_dataset, rank, world_size):
    settings = config["validation"]
    paths = settings["sample_files"]
    if isinstance(paths, str):
        paths = json.loads(Path(paths).read_text())
    dataset = M3Dataset(paths, history_dim=training_dataset.history_dim)
    if (dataset.history_source != training_dataset.history_source
            or dataset.history_capacity != training_dataset.history_capacity):
        raise ValueError("Validation history configuration differs from training")
    if len(dataset) != settings.get("expected_samples", len(dataset)):
        raise ValueError("Unexpected validation observation count")
    training_paths = {p.resolve() for p, _ in training_dataset.episodes}
    if training_paths & {p.resolve() for p, _ in dataset.episodes}:
        raise ValueError("Validation contains training episodes")
    indices = list(range(rank, len(dataset), world_size))
    size, workers = settings["batch_size"], settings.get("num_workers", 0)
    batches = [indices[i:i + size] for i in range(0, len(indices), size)]
    loader = DataLoader(
        dataset, batch_sampler=batches,
        collate_fn=Stage3Collator(processor, dataset.history_dim, config["max_sequence_length"],
                                 dataset.history_capacity),
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers else None,
        generator=torch.Generator().manual_seed(config["seed"] + 10000 + rank),
    )
    return dataset, loader, batches


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def summarize(values):
    loss_sum, count, absolute, squared, signed = values.tolist()
    return {"num_samples": int(count), "loss": loss_sum / count,
            "mae_pp": 100 * absolute / count, "rmse_pp": 100 * math.sqrt(squared / count),
            "bias_pp": 100 * signed / count}


@torch.no_grad()
def validate_during_training(model, dataset, loader, batches, output_dir, *, step, epoch,
                             epoch_actual, checkpoint_name, training_loss, rank, world_size):
    """Evaluate each held-out observation once, restore train/eval mode and all training RNGs."""
    device = next(model.parameters()).device
    output = Path(output_dir)
    directory = output / "validation" / f"step-{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted({group_name(summary) for _, summary in dataset.episodes})
    group_indices = [names.index(group_name(summary)) + 1 for _, summary in dataset.episodes]
    totals = torch.zeros((len(names) + 1, 5), dtype=torch.float64, device=device)
    if rank == 0:
        write_json(output / "validation-status.json", {"phase": "running", "step": step, "epoch": epoch})
        print(json.dumps({"kind": "validation_start", "step": step, "epoch": epoch}), flush=True)
    was_training = model.training
    rng_devices = [device.index] if device.type == "cuda" else []
    started, seen = perf_counter(), 0
    try:
        with torch.random.fork_rng(devices=rng_devices):
            model.eval()
            with (directory / f"predictions-rank{rank}.jsonl").open("w") as stream:
                for indices, batch in zip(batches, loader, strict=True):
                    inputs = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                              for key, value in batch["inputs"].items()}
                    target = batch["target_pt"].to(device)
                    result = model(inputs, target)
                    prediction, logits = result["pt"], result["bin_logits"]
                    targets = adjacent_bin_targets(target, logits.shape[-1])
                    losses = -(targets * logits.log_softmax(dim=-1)).sum(dim=-1).double()
                    if not torch.isfinite(prediction).all() or not torch.isfinite(losses).all():
                        raise ValueError("Nonfinite validation output")
                    error = (prediction - target).double()
                    values = torch.stack((losses, torch.ones_like(error), error.abs(), error.square(), error), dim=-1)
                    totals[0] += values.sum(dim=0)
                    groups = torch.tensor([group_indices[dataset.samples[i][0]] for i in indices], device=device)
                    for group in range(1, len(names) + 1):
                        totals[group] += values[groups == group].sum(dim=0)
                    for index, pred, pt in zip(indices, prediction.tolist(), target.tolist(), strict=True):
                        episode, sample, _ = dataset.samples[index]
                        row = {"index": index, "episode_id": str(dataset.episodes[episode][0]),
                               "group": names[group_indices[episode] - 1], "frame_index": sample["frame_index"],
                               "target_pt": pt, "predicted_pt": pred}
                        if "source_frame_index" in sample:
                            row["source_frame_index"] = sample["source_frame_index"]
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    seen += len(target)
                    if seen % 512 < len(target):
                        stream.flush()
                        write_json(directory / f"status-rank{rank}.json", {"samples": seen,
                                   "total": sum(map(len, batches)), "seconds": perf_counter() - started})
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = torch.tensor(perf_counter() - started, device=device, dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(totals)
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    finally:
        model.train(was_training)
    if int(totals[0, 1].item()) != len(dataset):
        raise ValueError("Validation coverage differs from the full dataset")
    metrics = {"kind": "validation", "step": step, "epoch": epoch, "epoch_actual": epoch_actual,
               "checkpoint": str(output / checkpoint_name), "train_loss_since_previous_validation": training_loss,
               **summarize(totals[0]), "groups": {name: summarize(totals[i + 1]) for i, name in enumerate(names)},
               "seconds": elapsed.item(), "world_size": world_size}
    if rank == 0:
        records = []
        for source in sorted(directory.glob("predictions-rank*.jsonl")):
            records.extend(json.loads(line) for line in source.read_text().splitlines())
        records.sort(key=lambda row: row["index"])
        if [row["index"] for row in records] != list(range(len(dataset))):
            raise ValueError("Duplicated or missing validation observations")
        with (directory / "predictions.jsonl").open("w") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(directory / "metrics.json", metrics)
        with (output / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        best = output / "best-checkpoint.json"
        if not best.exists() or metrics["loss"] < json.loads(best.read_text())["loss"]:
            write_json(best, {key: metrics[key] for key in ("checkpoint", "step", "epoch", "loss", "mae_pp")})
        write_json(output / "validation-status.json", {"phase": "completed", "step": step, "epoch": epoch})
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics
