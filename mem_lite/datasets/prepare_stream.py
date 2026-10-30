"""同步三视角视频和 PT CSV → 可直接训练的 StreamForest 样本。"""

import argparse
import csv
import json
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import torch

from mem_lite.datasets.m3_dataset import DATA_FORMAT, VIEW_ORDER
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder

DEFAULT_MEMORY_CONFIG = Path(__file__).parents[1] / "configs/streamforest.json"


def iter_observations(videos, progress, frame_stride=None, *, sample_fps=5.0,
                      start_frame=0, end_frame=None):
    """同步视角按 [start_frame,end_frame) 取单轮；保留原帧号和指定区间末帧。"""
    import av

    if len(videos) != 3 or (frame_stride is not None and frame_stride < 1) or sample_fps <= 0:
        raise ValueError("需要3个同步视角视频和正数采样频率/帧间隔")
    if start_frame < 0 or (end_frame is not None and end_frame <= start_frame):
        raise ValueError("单轮帧范围必须为非空的 [start_frame, end_frame)")
    with ExitStack() as stack:
        containers = [stack.enter_context(av.open(str(path))) for path in videos]
        rates = [float(video.streams.video[0].average_rate) for video in containers]
        if max(rates) - min(rates) > 1e-3:
            raise ValueError("三视角帧率不同，请先同步视频")
        counts = [video.streams.video[0].frames for video in containers]
        if all(counts) and len(set(counts)) != 1:
            raise ValueError("三视角帧数不同，请先同步视频")
        if end_frame is not None and any(count and end_frame > count for count in counts):
            raise ValueError("单轮帧范围超出源视频")
        frame_step = containers[0].streams.video[0].average_rate / Fraction(str(sample_fps))
        next_sample = 0
        streams = [video.decode(video=0) for video in containers]
        decoded = 0
        for index, frames in enumerate(zip(*streams, strict=True)):
            decoded = index + 1
            if index < start_frame:
                continue
            relative = index - start_frame
            # 选择达到采样时刻的第一帧；不能整除时也不累积固定 stride 的频率误差。
            selected = (relative % frame_stride == 0 if frame_stride is not None
                        else relative >= next_sample * frame_step)
            last = end_frame is not None and index == end_frame - 1
            if selected or last:
                yield {"frame_index": index, "images": [frame.to_image() for frame in frames], "pt": progress[index]}
                next_sample += 1
            if last:
                break
        if decoded <= start_frame or (end_frame is not None and decoded < end_frame):
            raise ValueError("单轮帧范围超出可解码视频")


def build_episode(observations, stream, output_dir, task_prompt, episode_id, history_source, *, source=None):
    """每条历史独立保存，避免把整段视频的全部快照留在内存。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for name in (*VIEW_ORDER, "history"):
        (output / name).mkdir()
    stream.reset()
    previous, count = None, 0
    with (output / "samples.jsonl").open("w", encoding="utf-8") as samples:
        for row in observations:
            frame_index = row["frame_index"]
            # PT 仅在落盘时保存，视觉编码和记忆维护不接收标签。
            history = stream.step(row["images"][0], frame_index, episode_id, task_prompt)
            image_paths = {}
            for view, image in zip(VIEW_ORDER, row["images"], strict=True):
                relative = f"{view}/{frame_index:08d}.png"
                image.save(output / relative)
                image_paths[view] = relative
            history_path = None
            if history is not None:
                history_path = f"history/{frame_index:08d}.pt"
                torch.save({"tokens": history.cpu(), "through_frame": previous}, output / history_path)
            samples.write(json.dumps({"frame_index": frame_index, "images": image_paths,
                                      "history": history_path, "pt": float(row["pt"])}, ensure_ascii=False) + "\n")
            previous, count = frame_index, count + 1
    if not count:
        raise ValueError("视频没有可用观测")
    summary = {"format_version": DATA_FORMAT, "episode_id": episode_id, "task_prompt": task_prompt,
               "num_samples": count, "history_dim": stream.encoder.hidden_size,
               "history_capacity": stream.memory.token_budget, "history_source": history_source}
    if source is not None:
        summary["source"] = source
    # 最后写 summary；中断产生的不完整目录不能被当作训练数据加载。
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs=3, required=True, metavar=("HIGH", "LEFT", "RIGHT"))
    parser.add_argument("--progress-csv", required=True)
    parser.add_argument("--task-prompt", required=True)
    parser.add_argument("--episode-id")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, help="单轮结束帧，不包含；指定时同时保留最后一帧")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-config", default=str(DEFAULT_MEMORY_CONFIG))
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--sample-fps", type=float, default=5.0, help="记忆和预测采样频率，默认 5 FPS")
    sampling.add_argument("--frame-stride", type=int, help="改用固定原始帧间隔采样")
    parser.add_argument("--max-pixels", type=int, default=65536)
    parser.add_argument("--min-pixels", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = json.loads(Path(args.memory_config).read_text())
    visual, processor, vision_hash = load_vision_encoder(args.model_path, args.device)
    stream = StreamMemory(QwenFrameEncoder(visual), processor, config, args.max_pixels, args.min_pixels)
    with Path(args.progress_csv).open() as source:
        progress = {int(row["frame_index"]): float(row["pt"]) for row in csv.DictReader(source)}
    sampling = {"frame_stride": args.frame_stride} if args.frame_stride is not None else {"sample_fps": args.sample_fps}
    origin = {"method": "streamforest-v1", "vision_sha256": vision_hash, "memory_config": config,
              **sampling, "max_pixels": args.max_pixels, "min_pixels": args.min_pixels}
    result = build_episode(iter_observations(args.videos, progress, **sampling,
                                            start_frame=args.start_frame, end_frame=args.end_frame), stream,
                           args.output_dir, args.task_prompt, args.episode_id or Path(args.videos[0]).stem, origin)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
