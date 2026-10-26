"""完整视频逐帧delta → 事件等权PT。只执行冻结ViT推理，不训练任何模型。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from mem_lite.datasets.m1_samples import read_annotation


def event_equal_progress(deltas, annotation, min_event_delta=1e-8):
    """整条视频的K个标注区间等权，包括最后一段，不根据描述筛除区间。

    对区间[a,b)，取锚点c=min(b,N-1)，累计delta[a+1:c+1]，
    使PT[a]=k/K、PT[c]=(k+1)/K。delta[a]属于上一段，不重复计入。
    普通段在下一段首帧达到锚点；末段在视频最后一帧达到1。
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    stages, frame_count = annotation["stages"], annotation["total_frames"]
    if deltas.shape != (frame_count,) or not np.isfinite(deltas).all() or np.any(deltas < 0):
        raise ValueError("delta需要与原视频逐帧对应，且为有限非负数")
    if not np.isfinite(min_event_delta) or min_event_delta < 0:
        raise ValueError("min_event_delta必须是有限非负数")
    if not stages or stages[0]["start_frame"] != 0 or stages[-1]["end_frame"] != frame_count:
        raise ValueError("至少需要一个区间，标注必须覆盖整条视频")
    for index, stage in enumerate(stages):
        if not 0 <= stage["start_frame"] < stage["end_frame"] <= frame_count:
            raise ValueError("状态区间为空或越界")
        if index and stages[index - 1]["end_frame"] != stage["start_frame"]:
            raise ValueError("状态区间必须连续且不重叠")
    if deltas[0] != 0:
        raise ValueError("首帧没有前一帧，delta[0]必须为0")
    event_count = len(stages)
    progress, events = np.empty(frame_count, dtype=np.float64), []
    for index, stage in enumerate(stages):
        start, end = stage["start_frame"], min(stage["end_frame"], frame_count - 1)
        changes = deltas[start + 1:end + 1]
        total = float(np.sum(changes, dtype=np.float64))
        if total <= min_event_delta:
            raise ValueError(f"事件{index + 1}（{start}→{end}）delta总和过小：{total}，需复核，不生成时间插值标签")
        progress[start] = index / event_count
        progress[start + 1:end + 1] = (index + np.cumsum(changes, dtype=np.float64) / total) / event_count
        progress[end] = (index + 1) / event_count
        events.append({
            "event_index": index + 1, "start_frame": start,
            "end_frame_exclusive": stage["end_frame"], "completion_frame": end,
            "delta_frame_range_inclusive": [start + 1, end], "delta_sum": total,
            "pt_start": index / event_count, "pt_complete": (index + 1) / event_count,
            "state_description": stage["state_description"],
        })
    return progress, events


def load_vision_encoder(model_path, device):
    """只读取Qwen3.5原始检查点的ViT/merger权重，不加载语言模型或LoRA。"""
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    model_path = Path(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True).vision_config
    # 只延迟分配参数；非持久RoPE缓冲必须按配置真实初始化，不能留在meta设备。
    with init_empty_weights(include_buffers=False):
        model = Qwen3_5VisionModel._from_config(config, attn_implementation="sdpa")
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted({name for key, name in weight_map.items() if key.startswith("model.visual.")})
    else:
        shards = ["model.safetensors"]
    state = {}
    digest = hashlib.sha256()
    for shard in shards:
        with safe_open(model_path / shard, framework="pt", device="cpu") as source:
            for key in sorted(source.keys()):
                if key.startswith("model.visual."):
                    tensor = source.get_tensor(key)
                    state[key.removeprefix("model.visual.")] = tensor
    # 对实际使用的视觉权重取摘要，不重复读取整个语言模型分片。
    for key, tensor in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    model.load_state_dict(state, assign=True)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model = model.to(device=device, dtype=dtype).requires_grad_(False).eval()
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    return model, processor, digest.hexdigest()


def encode_images(images, model, processor, max_pixels):
    """每帧分别做空间tokens平均池化；批次之间不混合，也不拼成视频tokens。"""
    import torch

    with torch.inference_mode():
        inputs = processor(images=images, return_tensors="pt", min_pixels=4096, max_pixels=max_pixels)
        parameter = next(model.parameters())
        grid = inputs["image_grid_thw"].to(parameter.device)
        pixels = inputs["pixel_values"].to(device=parameter.device, dtype=parameter.dtype)
        tokens = model(pixels, grid_thw=grid, return_dict=True).pooler_output.float()
        counts = (grid.prod(-1) // model.config.spatial_merge_size**2).tolist()
        return torch.stack([frame.mean(0) for frame in tokens.split(counts)]).cpu()


def extract_deltas(video_path, model, processor, batch_size=8, max_pixels=262144):
    """原始25/30FPS逐帧解码，保留全部帧，包括静止帧和任务完成后的画面。"""
    import av
    import torch

    if batch_size < 1:
        raise ValueError("batch_size必须大于0")
    batches, images, timestamps = [], [], []
    with av.open(str(video_path)) as video:
        stream = video.streams.video[0]
        stream.codec_context.thread_count = 2
        fps = float(stream.average_rate)
        for index, frame in enumerate(video.decode(stream)):
            timestamps.append(float(frame.pts * stream.time_base) if frame.pts is not None else index / fps)
            images.append(frame.to_image())
            if len(images) == batch_size:
                batches.append(encode_images(images, model, processor, max_pixels))
                images.clear()
        if images:
            batches.append(encode_images(images, model, processor, max_pixels))
    if not batches:
        raise ValueError("视频没有可解码帧")
    features = torch.cat(batches)
    deltas = torch.cat((torch.zeros(1), torch.linalg.vector_norm(features[1:] - features[:-1], dim=1)))
    return deltas.double().numpy(), {"fps": fps, "frame_count": len(features),
                                    "feature_dim": features.shape[1], "timestamps_seconds": timestamps}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_progress(annotation_path, video_path, model_path, output_dir, device="cuda",
                     batch_size=8, max_pixels=262144, min_event_delta=1e-8, vision_encoder=None):
    import torch
    import transformers

    annotation_path, video_path = Path(annotation_path).resolve(), Path(video_path).resolve()
    annotation = read_annotation(annotation_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    # 批量处理复用同一份冻结视觉编码器，避免每条视频重新加载权重。
    model, processor, vision_hash = vision_encoder if vision_encoder is not None else load_vision_encoder(model_path, device)
    deltas, info = extract_deltas(video_path, model, processor, batch_size, max_pixels)
    progress, events = event_equal_progress(deltas, annotation, min_event_delta)
    csv_path = output_dir / f"{video_path.stem}.progress.csv"
    with csv_path.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "delta", "pt"])
        writer.writerows((frame, float(delta), float(pt)) for frame, (delta, pt) in enumerate(zip(deltas, progress)))
    csv_path.with_suffix(".csv.tmp").replace(csv_path)
    metadata = {
        "format_version": "mem-lite-event-equal-progress-v2", "task_prompt": annotation["task_prompt"],
        "video_path": str(video_path), "video_sha256": file_sha256(video_path),
        "annotation_path": str(annotation_path), "annotation_sha256": file_sha256(annotation_path),
        "model_path": str(Path(model_path).resolve()), "vision_weights_sha256": vision_hash,
        "model_class": type(model).__name__, "model_dtype": str(next(model.parameters()).dtype),
        "feature": "mean of post-merger spatial tokens, FP32", "delta": "adjacent-frame raw L2, FP32",
        "rule": "all annotated intervals have equal weight, including the last; within-interval cumulative delta",
        "boundary_rule": "interval [a,b) uses delta[a+1:c+1], c=min(b,N-1); PT anchor at frame c",
        "batch_size": batch_size, "max_pixels": max_pixels,
        "min_event_delta": min_event_delta, "event_count": len(events), "events": events,
        "source": info, "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "csv_sha256": file_sha256(csv_path), "elapsed_seconds": perf_counter() - started,
    }
    (output_dir / f"{video_path.stem}.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return {"csv": str(csv_path), "frame_count": len(deltas), "event_count": len(events),
            "completed_at_frame": events[-1]["completion_frame"], "elapsed_seconds": metadata["elapsed_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="新建输出目录")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--min-event-delta", type=float, default=1e-8)
    args = parser.parse_args()
    print(json.dumps(prepare_progress(args.annotation, args.video, args.model_path, args.output_dir,
                                     args.device, args.batch_size, args.max_pixels, args.min_event_delta),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
