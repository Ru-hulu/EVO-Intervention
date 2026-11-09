"""Real-model forward/backward throughput probe; synthetic prefix is ONLY for sizing."""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoProcessor

from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.vst_dataset import VSTCollator
from mem_lite.models.vst_predictor import VSTCache, VSTProgressPredictor
from mem_lite.train_vst import device_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--history", type=int, default=6000)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpointing", choices=("all", "full_attention", "none"), default="all")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config["model"]["gradient_checkpointing"] = {"all": True, "full_attention": "full_attention", "none": False}[args.checkpointing]
    torch.manual_seed(42)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    processor = AutoProcessor.from_pretrained(config["model_path"], max_pixels=config["max_pixels"], local_files_only=True)
    collator = VSTCollator(processor, config["compression_ratio"])
    paths = json.loads(Path(config["sample_files"]).read_text())
    # Long task text and 64-token images stress the current-frame activation size.
    dataset = M3Dataset(paths, load_history=False)
    representative = max(range(len(dataset.episodes)), key=lambda e: len(dataset.episodes[e][1]["task_prompt"]))
    frame_indices = [i for i, (e, _, _) in enumerate(dataset.samples) if e == representative]
    items = [dataset[frame_indices[t % len(frame_indices)]] for _ in range(args.batch_size) for t in range(args.window_size)]
    batch = device_batch(collator(items, args.batch_size, args.window_size), "cuda")
    targets = torch.tensor([s["target_pt"] for s in items], device="cuda")
    model = VSTProgressPredictor(config["model_path"], **config["model"]).cuda().train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], fused=True)
    with torch.no_grad():
        initial = model(batch)["cache"]
        layers = {i: tuple(x.repeat(1, 1, (args.history + x.shape[2] - 1) // x.shape[2], 1)[:, :, :args.history].contiguous()
                           for x in pair) for i, pair in initial.layers.items()}
        prefix = VSTCache(layers, torch.ones(args.batch_size, args.history, dtype=torch.bool, device="cuda"))
    del initial
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); started = perf_counter()
        result = model(batch, targets, prefix)
        result["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        records.append({"step": step + 1, "seconds": perf_counter() - started, "loss": float(result["loss"].detach()),
            "grad_norm": float(norm), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})
        print(json.dumps(records[-1]), flush=True)
        del result
    report = {"batch_size": args.batch_size, "window_size": args.window_size, "history_tokens": args.history,
              "checkpointing": args.checkpointing,
              "sequence_length": batch["input_ids"].shape[1], "synthetic_prefix_for_sizing_only": True,
              "frames_per_second": args.batch_size * args.window_size / (sum(r["seconds"] for r in records[1:]) / len(records[1:])),
              "records": records}
    Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
