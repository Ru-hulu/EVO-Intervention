"""Measure online CLIP overhead without training or reporting task accuracy.

All three cases use the same Qwen weights, images and GPU. The controlled
clip_added case retains the original boundaries and discards the extra CLIP
descriptor. max_depth exercises the actual new memory path. Using an existing
minimum-similarity checkpoint here measures execution cost, not new-mode quality.
"""

import argparse
import gc
import json
import platform
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import transformers

from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.m3_inputs import M3Collator
from mem_lite.memory.clip import ClipBoundaryEncoder
from mem_lite.memory.splitting import override_memory_config
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder
from mem_lite.runtime import load_checkpoint, to_device


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": float(values.mean()), "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "min": float(values.min()), "max": float(values.max())}


def clean(device):
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def measure_clip(clip, images, device, baseline_bytes):
    for image in images[:8]:
        clip(image)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    total = []
    for image in images:
        started = perf_counter()
        clip(image)
        torch.cuda.synchronize(device)
        total.append(1000 * (perf_counter() - started))
    peak = torch.cuda.max_memory_allocated(device)
    preprocessing, forward, gpu = [], [], []
    for image in images:
        started = perf_counter()
        pixels = clip.processor(images=[image.convert("RGB")], return_tensors="pt")["pixel_values"]
        preprocessing.append(1000 * (perf_counter() - started))
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = perf_counter()
        start.record()
        with torch.autocast(device.type, enabled=False):
            result = clip.model(pixel_values=pixels.to(device=device, dtype=torch.float32))
        end.record()
        torch.cuda.synchronize(device)
        forward.append(1000 * (perf_counter() - started))
        gpu.append(start.elapsed_time(end))
        del result, pixels
    return {"total_ms": stats(total), "preprocessing_ms": stats(preprocessing),
            "transfer_and_forward_wall_ms": stats(forward), "transfer_and_forward_gpu_ms": stats(gpu),
            "parameter_count": sum(p.numel() for p in clip.model.parameters()),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in clip.model.parameters()),
            "peak_extra_allocated_bytes": peak - baseline_bytes,
            "provenance": clip.provenance}


@torch.no_grad()
def measure_pipeline(name, model, processor, config, source, observations, device, repeats, extra_clip=None):
    print(json.dumps({"phase": name, "stage": "initializing"}), flush=True)
    stream = StreamMemory(QwenFrameEncoder(model.qwen.model.visual), processor.image_processor,
                          source["memory_config"], source["max_pixels"], source["min_pixels"])
    bound_source = stream.bind_history_source(source)
    collator = M3Collator(processor, stream.encoder.hidden_size, config["max_sequence_length"],
                         stream.memory.token_budget)

    def step(row):
        history = stream.step(row["images"][0], row["frame_index"], row["episode_id"], row["task_prompt"])
        if extra_clip is not None:
            extra_clip(row["images"][0])
        inputs = collator([{"images": row["images"], "task_prompt": row["task_prompt"], "history_tokens": history}])
        result = model(to_device(inputs, device))
        return result, 0 if history is None else len(history)

    # Warm every measured shape/history length, including FLA/Triton compilation.
    for row in observations:
        step(row)
    stream.reset()
    clean(device)
    resident = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings, records = [], []
    print(json.dumps({"phase": name, "stage": "measuring", "resident_bytes": resident}), flush=True)
    for repeat in range(repeats):
        stream.reset()
        for row in observations:
            torch.cuda.synchronize(device)
            started = perf_counter()
            result, length = step(row)
            torch.cuda.synchronize(device)
            elapsed = 1000 * (perf_counter() - started)
            prediction = float(result["pt"][0])
            assert np.isfinite(prediction) and length <= stream.memory.token_budget
            timings.append(elapsed)
            records.append({"repeat": repeat, "frame_index": row["frame_index"], "ms": elapsed,
                            "history_length": length, "prediction_for_parity_only": prediction})
            del result
    result = {"latency_ms": stats(timings), "steady_after_16_frames_ms": stats([
        row["ms"] for i, row in enumerate(records) if i % len(observations) >= 16]),
        "resident_allocated_bytes": resident, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "records": records,
        "mean_history_length": float(np.mean([row["history_length"] for row in records])),
        "history_source": bound_source}
    stream.reset()
    del stream, collator
    clean(device)
    print(json.dumps({"phase": name, "stage": "done", "latency_ms": result["latency_ms"],
                      "peak_allocated_bytes": result["peak_allocated_bytes"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--clip-model-path", required=True)
    parser.add_argument("--sample-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.frames <= 16 or args.repeats < 1:
        parser.error("需要至少 17 帧和一次测量")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(args.threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dataset = M3Dataset([args.sample_file], load_history=False)
    if len(dataset.episodes) != 1 or len(dataset) < args.frames:
        raise ValueError("需要一条包含足够连续观测的 episode")
    observations = [dataset[i] for i in range(args.frames)]
    model, processor, config = load_checkpoint(args.checkpoint, args.device, args.model_path,
                                               event_split_method="min_similarity")
    source = deepcopy(config["history_source"])
    report = {"protocol": vars(args), "gpu": torch.cuda.get_device_name(device), "batch_size": 1,
              "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "excludes_model_load_disk_io_and_warmup": True, "accuracy_evaluation": False}

    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    report["baseline"] = measure_pipeline("baseline", model, processor, config, source, observations, device, args.repeats)
    save()
    clean(device)
    before = torch.cuda.memory_allocated(device)
    clip = ClipBoundaryEncoder(args.clip_model_path, device)
    clean(device)
    report["clip_resident_extra_bytes"] = torch.cuda.memory_allocated(device) - before
    report["clip_only"] = measure_clip(clip, [row["images"][0] for row in observations], device, before)
    report["clip_added"] = measure_pipeline("clip_added", model, processor, config, source, observations,
                                            device, args.repeats, extra_clip=clip)
    a, b = report["baseline"]["records"], report["clip_added"]["records"]
    assert [r["history_length"] for r in a] == [r["history_length"] for r in b]
    error = max(abs(x["prediction_for_parity_only"] - y["prediction_for_parity_only"]) for x, y in zip(a, b, strict=True))
    assert error < 1e-5
    report["control_max_prediction_difference"] = error
    save()
    del clip
    clean(device)
    source["memory_config"] = override_memory_config(source["memory_config"], "max_depth", args.clip_model_path)
    report["max_depth"] = measure_pipeline("max_depth", model, processor, config, source, observations, device, args.repeats)
    base, added, depth = [report[name]["latency_ms"]["mean"] for name in ("baseline", "clip_added", "max_depth")]
    report["summary"] = {"controlled_extra_ms": added - base, "controlled_extra_percent": 100 * (added / base - 1),
                         "actual_mode_extra_ms": depth - base, "actual_mode_extra_percent": 100 * (depth / base - 1),
                         "peak_extra_allocated_bytes": report["max_depth"]["peak_allocated_bytes"] - report["baseline"]["peak_allocated_bytes"]}
    report["completed"] = True
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
