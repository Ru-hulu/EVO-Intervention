"""离线构造 M2 扰动事件队列和全部前缀监督，不加载任何模型。"""

import argparse
import json
from pathlib import Path

from mem_lite.datasets.m1_samples import read_annotation
from mem_lite.datasets.m2_samples import build_queue_samples
from mem_lite.datasets.video import extract_observations, save_missing_frames

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m2_data.json"


def prepare_m2(annotation_path, output, config, video_path=None):
    annotation_path, output = Path(annotation_path).resolve(), Path(output).resolve()
    annotation = read_annotation(annotation_path)
    video_path = (Path(video_path).resolve() if video_path
                  else (annotation_path.parent.parent / annotation["video_file"]).resolve())
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到主视角视频：{video_path}，可通过 --video 指定")
    output.mkdir(parents=True, exist_ok=False)
    boundaries = {frame for stage in annotation["stages"] for frame in (stage["start_frame"], stage["end_frame"] - 1)}
    info = extract_observations(video_path, boundaries, output / "frames", config)
    if info["frame_count"] != annotation["total_frames"]:
        raise ValueError("视频实际帧数与标注不一致")
    samples = build_queue_samples(
        annotation, info["kept_frames"], info["timestamps"], config["draws_per_video"], config["seed"],
    )
    required = {frame["frame_index"] for sample in samples for event in sample["inputs"]["events"]
                for frame in event["frames"]}
    save_missing_frames(video_path, required, output / "frames")
    with (output / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for sample in samples:
            sample["sample_id"] = f"{video_path.stem}-{sample['sample_id']}"
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    summary = {
        "format_version": "mem-lite-m2-v1", "annotation_path": str(annotation_path),
        "main_video_path": str(video_path), "config": config, "source_fps": info["fps"],
        "source_frame_count": info["frame_count"], "queue_count": len(samples),
        "prefix_count": sum(len(sample["targets"]) for sample in samples),
        "base_queue_count": len(info["kept_frames"]),
        "sample_boundary_additions": len(required - set(info["kept_frames"])),
    }
    for filename, value in (("annotation.json", annotation), ("summary.json", summary),
                            ("observations.json", info["observations"])):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="新建输出目录")
    parser.add_argument("--video", type=Path, help="主视角MP4，默认读取标注的video_file")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(prepare_m2(args.annotation, args.output, config, args.video), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
