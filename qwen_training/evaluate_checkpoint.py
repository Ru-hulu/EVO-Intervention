"""Legacy MEM checkpoint evaluation on a saved held-out video split.

Launch independent GPU shards with torchrun; no gradients or DDP synchronization is needed.
Every sample is evaluated under both prompt modes, reusing its visual encoding.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader

from qwen_training.dataset import VLMProgressDataset, collate_fn
from qwen_training.model import QwenProgressModel
from qwen_training.prompts import build_messages
from qwen_training.train_distributed import phase_batches


METADATA = ("sample_id", "dataset", "video_id", "task_index", "episode", "sample_type",
            "sampling_method", "frame_indices", "input_state", "target")


class EvaluationDataset(VLMProgressDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["metadata"] = {key: self.samples[index][key] for key in METADATA}
        return sample


def collate_evaluation(samples):
    inputs, targets = collate_fn(samples)
    return inputs, targets, [sample["metadata"] for sample in samples]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_samples(samples, split, limit=0):
    train_videos = {tuple(value) for value in split["train"]}
    validation_videos = {tuple(value) for value in split["validation"]}
    if train_videos & validation_videos:
        raise ValueError("Training and validation video identities overlap")
    selected = [s for s in samples if (s["dataset"], s["video_id"]) in validation_videos]
    if {(s["dataset"], s["video_id"]) for s in selected} != validation_videos:
        raise ValueError("Validation videos missing from index")
    if len({s["sample_id"] for s in selected}) != len(selected):
        raise ValueError("Duplicate sample IDs")
    if limit:
        # Coverage of datasets, current/previous stages and window types in the smoke run.
        groups = {}
        for sample in selected:
            key = (sample["dataset"], sample["input_state"]["nowid"], sample["target"]["nowid"], sample["sample_type"])
            groups.setdefault(key, []).append(sample)
        priority = [values[0] for _, values in sorted(groups.items())]
        chosen = {s["sample_id"] for s in priority[:limit]}
        for sample in selected:
            if len(chosen) >= limit:
                break
            chosen.add(sample["sample_id"])
        selected = [s for s in selected if s["sample_id"] in chosen]
    return selected


def load_model(checkpoint, device):
    if checkpoint.get("config", {}).get("progress_format", "").startswith("reference_window_"):
        raise ValueError("六行参考轨迹模型请使用 qwen_training.infer_reference")
    if "progress_head" in checkpoint:
        raise ValueError("旧回归头检查点需使用旧版代码评估；当前模型从文本解析进度")
    config = checkpoint["config"]
    model = QwenProgressModel(
        config["model_path"], dtype=torch.bfloat16, freeze_backbone=True,
        max_pixels=config["max_pixels"], lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"])
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.visual_adapter.load_state_dict(checkpoint["visual_adapter"], strict=True)
    expected = get_peft_model_state_dict(model.qwen, save_embedding_layers=False)
    if expected.keys() != checkpoint["lora"].keys():
        raise ValueError("Checkpoint LoRA keys do not match model")
    for name, value in expected.items():
        if value.shape != checkpoint["lora"][name].shape:
            raise ValueError(f"Checkpoint LoRA shape mismatch: {name}")
    loaded = set_peft_model_state_dict(model.qwen, checkpoint["lora"])
    if loaded.unexpected_keys:
        raise ValueError(loaded.unexpected_keys)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stop-result", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    stopped = json.loads(args.stop_result.read_text())
    if stopped["status"] != "stopped_after_verified_epoch04":
        raise RuntimeError("Evaluation must wait for the authorized checkpoint validation and training stop")
    if args.checkpoint.resolve() != Path(stopped["verification"]["checkpoint"]).resolve():
        raise ValueError("Checkpoint does not match the verified epoch-04 checkpoint")
    digest = sha256(args.checkpoint)
    if digest != stopped["verification"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash changed after validation")
    rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
    if not is_fast_path_available:
        raise RuntimeError("Required fast kernels are unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    dataset = EvaluationDataset(config["index_files"])
    split = json.loads(args.split.read_text())
    selected = select_samples(dataset.samples, split, args.limit)
    dataset.samples = selected[rank::world_size]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / f"predictions-rank{rank:02d}.jsonl"
    if predictions_path.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {predictions_path}")
    manifest = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=digest,
                    progress_format=config.get("progress_format", "regression"),
                    checkpoint_epoch=checkpoint["epoch"], checkpoint_step=checkpoint["step"],
                    split_file=str(args.split), split_sha256=sha256(args.split),
                    index_sha256={p: sha256(p) for p in config["index_files"]},
                    split=split, evaluation_split="validation", independent_test_set=False,
                    samples=len(selected), sample_ids=[s["sample_id"] for s in selected],
                    modes=["short", "context"], generation="greedy", max_new_tokens=args.max_new_tokens,
                    batch_size_per_gpu=args.batch_size, world_size=world_size,
                    previous_state_source="ground_truth_previous_observation_not_model_rollout",
                    smoke_test=bool(args.limit), started_unix=time.time(),
                    model_source_sha256=sha256(Path(__file__).with_name("model.py")),
                    evaluator_source_sha256=sha256(__file__))
    if rank == 0:
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    del selected
    model = load_model(checkpoint, device)
    del checkpoint
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_evaluation,
                        pin_memory=True, persistent_workers=False,
                        multiprocessing_context="spawn" if args.num_workers else None,
                        prefetch_factor=2 if args.num_workers else None)
    processed = 0
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    with predictions_path.open("x") as output, torch.inference_mode():
        for inputs, targets, metadata in phase_batches(loader):
            images = inputs["images"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                visual = model.encode_history(images, inputs["frame_mask"])
                for mode, probability in [("short", 1.0), ("context", 0.0)]:
                    messages = build_messages(inputs["input_states"], state_drop_probability=probability)
                    predictions = model.generate(messages, **visual, max_new_tokens=args.max_new_tokens)
                    values = predictions["progress"].float().cpu().tolist()
                    for meta, target_text, response, progress in zip(metadata, targets["responses"], predictions["responses"], values, strict=True):
                        progress_valid = math.isfinite(progress)
                        row = dict(meta, mode=mode, target_response=target_text,
                                   response=response, predicted_progress=progress if progress_valid else None,
                                   progress_parse_valid=progress_valid,
                                   generated_tokens_reencoded=len(model.tokenizer.encode(response, add_special_tokens=False)))
                        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            processed += len(metadata)
            if processed % (args.batch_size * 10) == 0 or processed == len(dataset):
                print(json.dumps(dict(rank=rank, samples=processed, total=len(dataset),
                      elapsed_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)), flush=True)
    completion = dict(rank=rank, status="complete", samples=processed, responses=processed*2,
                      seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                      predictions_sha256=sha256(predictions_path), finished_unix=time.time())
    (args.output_dir / f"complete-rank{rank:02d}.json").write_text(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
