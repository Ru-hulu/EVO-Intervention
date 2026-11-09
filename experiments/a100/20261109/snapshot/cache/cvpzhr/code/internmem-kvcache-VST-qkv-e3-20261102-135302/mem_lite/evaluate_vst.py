"""Complete chronological validation with the trained VST cache and no gradients."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from mem_lite.datasets.vst_dataset import EpisodeWindows, WindowCollator
from mem_lite.models.vst_predictor import VSTProgressPredictor
from mem_lite.train_vst import CHECKPOINT_FORMAT, device_batch, load_trainable_state, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-files")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved["format"] != CHECKPOINT_FORMAT:
            raise ValueError("Expected a VST checkpoint")
        config = saved["config"]
        model = VSTProgressPredictor(config["model_path"], **config["model"]).to(device).eval()
        load_trainable_state(model, saved["model"])
        del saved
        paths = json.loads(Path(args.sample_files or config["validation_sample_files"]).read_text())
        dataset = EpisodeWindows(paths, config["batch_size"], config["window_size"], rank, world)
        processor = AutoProcessor.from_pretrained(config["model_path"], max_pixels=config["max_pixels"], local_files_only=True)
        loader = DataLoader(dataset, batch_size=None,
            collate_fn=WindowCollator(processor, config["compression_ratio"], config["max_sequence_length"]),
            num_workers=config["num_workers"], pin_memory=True,
            multiprocessing_context="spawn" if config["num_workers"] else None)
        cache, totals = None, torch.zeros(4, device=device, dtype=torch.float64)
        began = perf_counter()
        with torch.inference_mode():
            for item in loader:
                if item["reset"]:
                    cache = None
                inputs = device_batch(item["inputs"], device)
                output = model(inputs, cache=cache)
                cache = output["cache"]
                target = item["target_pt"].to(device)
                error = (output["pt"] - target).double()[inputs["frame_valid"]]
                totals += torch.stack((error.new_tensor(error.numel()), error.abs().sum(), error.square().sum(), error.sum()))
        if world > 1:
            dist.all_reduce(totals)
        if int(totals[0]) != len(dataset.frames):
            raise RuntimeError("Validation coverage mismatch")
        if rank == 0:
            directory = Path(args.output_dir); directory.mkdir(parents=True, exist_ok=True)
            report = {"num_samples": int(totals[0]), "mae_pp": float(100 * totals[1] / totals[0]),
                "rmse_pp": float(100 * (totals[2] / totals[0]).sqrt()), "bias_pp": float(100 * totals[3] / totals[0]),
                "seconds": perf_counter() - began, "checkpoint": args.checkpoint,
                "compression_ratio": config["compression_ratio"], "data_signature": dataset.signature}
            write_json(directory / "metrics.json", report)
            print(json.dumps(report), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
