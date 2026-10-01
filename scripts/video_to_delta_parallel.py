#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""多个视频进程并行，逐帧调用原 QwenDelta，输出完整的新版阶段描述。"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import json
import multiprocessing
from pathlib import Path
import time

import av
from delta_to_progress import frame_descriptions
import torch
from video_to_delta import QwenDelta

encoder = None


def initialize_worker(model_dir):
    global encoder  # noqa: PLW0603 — 每个独立进程只加载一次编码器。
    torch.set_num_threads(2)
    encoder = QwenDelta(model_dir)


def process_video(video_path, label_path, output_path):
    started = time.monotonic()
    annotation = json.loads(label_path.read_text(encoding="utf-8-sig"))
    deltas = []
    previous = None
    with av.open(str(video_path)) as video:
        for frame in video.decode(video=0):
            current = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            delta = 0.0 if previous is None else encoder.image2delta(previous, current)
            deltas.append(delta)
            previous = current

    descriptions = frame_descriptions(annotation, len(deltas))
    temporary = output_path.with_suffix(".txt.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for value, description in zip(deltas, descriptions, strict=True):
            output.write(f"{value:.6f} {description}\n")
    temporary.replace(output_path)
    elapsed = time.monotonic() - started
    return {"file": video_path.stem, "frames": len(deltas), "seconds": round(elapsed, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("/root/video"))
    parser.add_argument("--label-dir", type=Path, default=Path("/root/lable"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/pt"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/root/autodl-tmp/qwen3-vl-ft/models/Qwen3-VL-4B-Instruct")
    )
    parser.add_argument("--file-list", type=Path, required=True, help="每行一个视频文件名")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(args.model_dir,),
    ) as pool:
        futures = []
        for name in args.file_list.read_text().splitlines():
            video = args.video_dir / name
            output = args.output_dir / f"{video.stem}-delta.txt"
            if output.exists():
                print(f"已有结果，跳过 {output.name}", flush=True)
                continue
            futures.append(pool.submit(process_video, video, args.label_dir / f"{video.stem}.JSON", output))
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
