"""评估缓存或在线重建历史：进度误差、逐帧延迟和 CUDA 显存峰值。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from mem_lite.datasets.m3_dataset import M3Dataset
from mem_lite.datasets.m3_inputs import M3Collator
from mem_lite.runtime import StreamingProgressPredictor, load_checkpoint, to_device


@torch.no_grad()
def evaluate(model, processor, dataset, max_sequence_length=8192, online=False, without_history=False):
    model.eval()
    device = next(model.parameters()).device
    collator = M3Collator(processor, dataset.history_dim, max_sequence_length, dataset.history_capacity)
    runner = (StreamingProgressPredictor(model, processor, dataset.history_source, max_sequence_length)
              if online and not without_history else None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    for sample in dataset:
        # 读图/缓存 IO 不计入模型链路延迟；在线模式包括主视角编码和记忆维护。
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = perf_counter()
        if runner is not None:
            result = runner(sample["images"], sample["frame_index"], sample["episode_id"], sample["task_prompt"])
            history_length = result["history_length"]
        else:
            if without_history:
                sample = {**sample, "history_tokens": None}
            inputs = collator([sample])
            result = model(to_device(inputs, device))
            history_length = int(inputs["history_valid_mask"].sum())
        prediction = float(result["pt"][0])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = 1000 * (perf_counter() - started)
        records.append({"episode_id": sample["episode_id"], "frame_index": sample["frame_index"],
                        "target_pt": sample["target_pt"], "predicted_pt": prediction,
                        "history_length": history_length, "latency_ms": elapsed})
    errors = np.array([row["predicted_pt"] - row["target_pt"] for row in records])
    times = np.array([row["latency_ms"] for row in records])
    steady = times[1:] if len(times) > 1 else times
    metrics = {
        "mode": "without_history" if without_history else "online" if online else "cached",
        "num_samples": len(records), "mae_pp": float(100 * np.abs(errors).mean()),
        "rmse_pp": float(100 * np.sqrt(np.square(errors).mean())), "bias_pp": float(100 * errors.mean()),
        "first_frame_ms": float(times[0]), "latency_ms_mean": float(steady.mean()),
        "latency_ms_p50": float(np.percentile(steady, 50)), "latency_ms_p95": float(np.percentile(steady, 95)),
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
    return metrics, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", help="迁移机器时覆盖同一基座的本地路径")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--online", action="store_true", help="按顺序重建记忆，不加载历史快照")
    parser.add_argument("--without-history", action="store_true", help="对同一检查点做移除历史的消融")
    args = parser.parse_args()
    model, processor, config = load_checkpoint(args.checkpoint, args.device, args.model_path)
    dataset = M3Dataset(args.sample_files, config["model"]["history_dim"],
                        load_history=not (args.online or args.without_history))
    if dataset.history_source != config["history_source"] or dataset.history_capacity != config["history_capacity"]:
        raise ValueError("评估集的记忆配置与检查点不一致")
    metrics, records = evaluate(model, processor, dataset, config["max_sequence_length"],
                                args.online, args.without_history)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
