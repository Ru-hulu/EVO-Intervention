"""从一个主视角视频与阶段标注构造 M1 训练样本。

python -m mem_lite.datasets.prepare_m1 --annotation ... --output ...
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from mem_lite.datasets.m1_samples import build_sample, read_annotation, sample_windows
from mem_lite.datasets.preview import write_preview
from mem_lite.datasets.video import extract_observations, save_missing_frames

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m1_data.json"


def prepare_m1(annotation_path, output, config, video_path=None, expert_tokens_path=None, main_view="main"):
    annotation_path, output = Path(annotation_path).resolve(), Path(output).resolve()
    annotation = read_annotation(annotation_path)
    video_path = (
        Path(video_path).resolve()
        if video_path
        else (annotation_path.parent.parent / annotation["video_file"]).resolve()
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到主视角视频：{video_path}，可通过 --video 指定")
    if config["difference_width"] <= 0 or not 0 <= config["min_changed_fraction"] <= 1:
        raise ValueError("difference_width 需为正数，min_changed_fraction 需位于 [0,1]")
    output.mkdir(parents=True, exist_ok=False)
    boundaries = {frame for stage in annotation["stages"] for frame in (stage["start_frame"], stage["end_frame"] - 1)}
    video_info = extract_observations(video_path, boundaries, output / "frames", config)
    if video_info["frame_count"] != annotation["total_frames"]:
        raise ValueError(f"视频实际 {video_info['frame_count']} 帧，与标注 {annotation['total_frames']} 帧不一致")

    samples, skipped = [], []
    expert_path = str(Path(expert_tokens_path).resolve()) if expert_tokens_path else None
    for window in sample_windows(annotation, config):
        sample = build_sample(
            window, video_info["kept_frames"], video_info["timestamps"], annotation["task_prompt"], expert_path
        )
        if sample is None:
            skipped.append(window)
        else:
            sample["sample_id"] = f"{video_path.stem}-{sample['sample_id']}"
            sample["inputs"]["main_view"] = main_view
            samples.append(sample)
    required_frames = {frame["frame_index"] for sample in samples for frame in sample["inputs"]["frames"]}
    save_missing_frames(video_path, required_frames, output / "frames")

    counts = Counter((sample["metadata"]["case"], sample["target"]["valid"]) for sample in samples)
    summary = {
        "format_version": "mem-lite-m1-v1",
        "annotation_path": str(annotation_path),
        "main_video_path": str(video_path),
        "main_view": main_view,
        "expert_tokens_path": expert_path,
        "config": config,
        "source_fps": video_info["fps"],
        "source_frame_count": video_info["frame_count"],
        "candidate_count": sum(row["candidate_10fps"] for row in video_info["observations"]),
        "static_removed": sum(not row["kept"] for row in video_info["observations"]),
        "base_queue_count": len(video_info["kept_frames"]),
        "sample_boundary_additions": len(required_frames - set(video_info["kept_frames"])),
        "sample_count": len(samples),
        "skipped_empty": len(skipped),
        "by_case": {case: {"valid": counts[case, 1], "invalid": counts[case, 0]} for case in config["samples_per_case"]},
    }
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    for filename, value in [
        ("annotation.json", annotation),
        ("summary.json", summary),
        ("observations.json", video_info["observations"]),
        ("skipped.json", skipped),
    ]:
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_preview(output, annotation, video_info, samples, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="新建输出目录，避免覆盖已有样本")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", type=Path, help="主视角 MP4；默认从 annotations 的上级目录读取 video_file")
    parser.add_argument("--main-view", default="main", help="主视角名称，例如 front/top")
    parser.add_argument("--expert-tokens", type=Path, help="已有专家 tokens 文件；未提供时仅预留 null 接口")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = prepare_m1(args.annotation, args.output, config, args.video, args.expert_tokens, args.main_view)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"样本预览：{args.output.resolve() / 'preview.html'}")


if __name__ == "__main__":
    main()
