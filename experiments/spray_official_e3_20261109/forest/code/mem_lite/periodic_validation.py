"""Periodic cached validation for the original Forest model."""
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator

SUITES = ("Count", "Order", "State")


def validation_update_points(num_batches, accumulation, interval):
    """Round each fractional-epoch boundary up to a completed optimizer update."""
    if not 0 < interval <= 1:
        raise ValueError("validation_interval_epochs must be in (0,1]")
    parts = round(1 / interval)
    if not math.isclose(parts * interval, 1):
        raise ValueError("validation_interval_epochs must divide one epoch exactly")
    updates = math.ceil(num_batches / accumulation)
    return {math.ceil(updates * part / parts) for part in range(1, parts + 1)}


class ValidationCollator:
    def __init__(self, collator, suite_by_episode):
        self.collator = collator
        self.suite_by_episode = suite_by_episode

    def __call__(self, samples):
        batch = self.collator(samples)
        batch["suite_indices"] = torch.tensor([
            SUITES.index(self.suite_by_episode[row["episode_id"]]) for row in samples
        ], dtype=torch.long)
        return batch


def make_validation_loader(config, processor, train_dataset, rank, world_size):
    dataset = M3Dataset(config["validation_sample_files"], train_dataset.history_dim)
    for name in ("history_source", "history_capacity", "history_dim"):
        if getattr(dataset, name) != getattr(train_dataset, name):
            raise ValueError(f"Validation {name} differs from training")
    signature = config.get("validation_data_signature")
    if signature is not None and signature != dataset.signature:
        raise ValueError("Validation data changed since the saved checkpoint")
    config["validation_data_signature"] = dataset.signature
    collator = ValidationCollator(
        Stage3Collator(processor, dataset.history_dim, config["max_sequence_length"],
                       dataset.history_capacity),
        config["validation_suite_by_episode"],
    )
    workers = config.get("validation_num_workers", 0)
    # No padding, shuffling or duplicated tail samples; uneven ranks reduce only at the end.
    return DataLoader(
        dataset, batch_size=config.get("validation_batch_size", config["batch_size"]),
        sampler=range(rank, len(dataset), world_size), collate_fn=collator,
        num_workers=workers, pin_memory=True,
        multiprocessing_context="spawn" if workers else None,
        generator=torch.Generator().manual_seed(config["seed"] + rank),
    )


def metrics_from_totals(totals):
    def metrics(row):
        absolute, squared, signed, count = row.tolist()
        if count == 0:
            return {"num_samples": 0, "mae_pp": None, "rmse_pp": None, "bias_pp": None}
        return {
            "num_samples": int(count), "mae_pp": 100 * absolute / count,
            "rmse_pp": 100 * math.sqrt(squared / count), "bias_pp": 100 * signed / count,
        }
    return {
        **metrics(totals.sum(0)),
        "by_suite": {suite: metrics(totals[i]) for i, suite in enumerate(SUITES)},
    }


@torch.inference_mode()
def validate(model, loader, device):
    """Use the unwrapped model: no per-forward DDP collectives on uneven shards."""
    totals = torch.zeros((len(SUITES), 4), device=device, dtype=torch.float64)
    training = model.training
    model.eval()
    try:
        # Evaluation and iterator construction must not advance training dropout RNG.
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            for batch in loader:
                inputs = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                          for k, v in batch["inputs"].items()}
                target = batch["target_pt"].to(device, non_blocking=True)
                error = (model(inputs)["pt"] - target).double()
                if not torch.isfinite(error).all():
                    raise ValueError("Non-finite validation predictions")
                suite = batch["suite_indices"].to(device, non_blocking=True)
                totals.index_add_(0, suite, torch.stack(
                    (error.abs(), error.square(), error, torch.ones_like(error)), dim=1,
                ))
            if dist.is_initialized():
                dist.all_reduce(totals)
    finally:
        model.train(training)
    result = metrics_from_totals(totals.cpu())
    if result["num_samples"] != len(loader.dataset):
        raise ValueError("Validation must count each cached observation exactly once")
    return result
