"""按标注拆分常规成功 episode，生成六时刻窗口索引；视频在训练取样时解码。

python -m qwen_training.build_reference_index --dataset-root /path/dataset \
    --reference-selection /path/selection.json --output-dir /path/dataset/reference_windows
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil

from qwen_training.delta_to_progress import episode_to_progress
from qwen_training.reference_io import FORMAT_VERSION
from qwen_training.reference_sampling import make_frame_records, sample_episode_frames


CAMERAS = {
    "top": "observation.images.camera_top",
    "left": "observation.images.camera_wrist_left",
    "right": "observation.images.camera_wrist_right",
}
TASK = "将螺钉插入套筒，并将装配好的组件放入盒子。"


def build_reference_index(*, dataset_name, label_dir, delta_dir, video_dirs, reference_selection,
                          output_dir, task=TASK, video_ids=None, samples_per_episode=32,
                          terminal_samples=2, seed=0, reference_dataset=None):
    """普通窗口在整个 episode 均匀选当前帧；额外补采末帧窗口提供 done 正例。"""
    if samples_per_episode < 1 or terminal_samples < 0:
        raise ValueError("随机窗口数至少为1，末帧补采数不能为负")
    output_dir = Path(output_dir).resolve()
    label_dir, delta_dir = Path(label_dir), Path(delta_dir)
    selection_path = Path(reference_selection).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if len(selection["reference_images"]) != 5:
        raise ValueError("专家轨迹必须包含五张图片")
    label_files = sorted(p for p in label_dir.iterdir()
                         if p.suffix.lower() == ".json" and (video_ids is None or p.stem in video_ids))
    if not label_files:
        raise FileNotFoundError(f"没有匹配的标注：{label_dir}")
    # 开始写索引前确认三视角都可读取，不用空白图替代缺失的真实相机。
    for label_path in label_files:
        for directory in video_dirs.values():
            path = Path(directory) / f"{label_path.stem}.mp4"
            if not path.is_file():
                raise FileNotFoundError(path)

    references = output_dir / "references"
    references.mkdir(parents=True, exist_ok=True)
    reference_images = []
    for i, source in enumerate(selection["reference_images"], 1):
        source = selection_path.parent / source
        destination = references / f"reference-{i:03d}{source.suffix}"
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        reference_images.append(str(destination.relative_to(output_dir)))
    copied_selection = dict(selection, reference_images=[Path(p).name for p in reference_images])
    if "frames" in selection:
        copied_selection["frames"] = [dict(frame, image=Path(image).name)
                                      for frame, image in zip(selection["frames"], reference_images)]
    (references / "selection.json").write_text(
        json.dumps(copied_selection, ensure_ascii=False, indent=2), encoding="utf-8")
    reference_source = {
        "dataset": reference_dataset or dataset_name,
        "video_id": Path(selection["source_video"]).stem,
    }
    rng = random.Random(seed)
    episode_count = window_count = 0
    windows_tmp = output_dir / "windows.jsonl.tmp"
    episodes_tmp = output_dir / "episodes.jsonl.tmp"
    with windows_tmp.open("w", encoding="utf-8") as windows, episodes_tmp.open("w", encoding="utf-8") as episodes:
        for label_path in label_files:
            annotation = json.loads(label_path.read_text(encoding="utf-8"))
            delta_path = delta_dir / f"{label_path.stem}-delta.txt"
            deltas = [float(line.split()[0]) for line in delta_path.read_text().splitlines() if line.strip()]
            task_ids = [stage["id"] for stage in annotation["taskid"]]
            videos = {view: os.path.relpath((Path(directory) / f"{label_path.stem}.mp4").resolve(), output_dir)
                      for view, directory in video_dirs.items()}
            for task_index, annotated_task in enumerate(annotation["tasks"]):
                for episode in annotated_task["episodes"]:
                    subtasks = episode["subtasks"]
                    start, end = subtasks[0]["30FPSstart_frame"], subtasks[-1]["30FPSend_frame"]
                    progress, _ = episode_to_progress(deltas, subtasks, task_ids)
                    metadata = {
                        "dataset": dataset_name, "video_id": label_path.stem,
                        "task_index": task_index, "episode": episode["episode"],
                        "episode_start_frame": start, "episode_end_frame": end,
                        "episode_start_time": start / 30, "episode_end_time": end / 30,
                        "fps": 30, "end_inclusive": True,
                        "done_policy": "successful_episode_last_frame",
                    }
                    episodes.write(json.dumps({
                        **metadata, "subtasks": subtasks,
                        "annotation": os.path.relpath(label_path.resolve(), output_dir),
                        "delta": os.path.relpath(delta_path.resolve(), output_dir),
                    }, ensure_ascii=False) + "\n")
                    for kind, count in (("uniform", samples_per_episode), ("terminal", terminal_samples)):
                        for i in range(count):
                            indices = sample_episode_frames(start, end, rng, current=end if kind == "terminal" else None)
                            frames, targets = make_frame_records(indices, progress, start, end)
                            record = {
                                "format_version": FORMAT_VERSION, **metadata,
                                "sample_id": f"{dataset_name}/{label_path.stem}/t{task_index}-e{episode['episode']}/{kind}-{i:04d}",
                                "sample_type": kind, "task": task, "video_paths": videos,
                                "reference_images": reference_images, "reference_source": reference_source,
                                "frames": frames, "targets": targets,
                            }
                            windows.write(json.dumps(record, ensure_ascii=False) + "\n")
                            window_count += 1
                    episode_count += 1
    windows_tmp.replace(output_dir / "windows.jsonl")
    episodes_tmp.replace(output_dir / "episodes.jsonl")
    manifest = {
        "format_version": FORMAT_VERSION, "dataset": dataset_name, "videos": len(label_files),
        "episodes": episode_count, "windows": window_count, "seed": seed,
        "samples_per_episode": samples_per_episode, "terminal_samples": terminal_samples,
        "interval_seconds": [0.5, 1.0], "sampling_grid_fps": 30,
        "reference_source": reference_source,
        "done_policy": "successful_episode_last_frame",
        "shuffle": "训练读取时以0.3概率交换有效帧的前后两段；验证不交换",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--label-dir", type=Path, help="默认 dataset-root/labels")
    parser.add_argument("--delta-dir", type=Path, help="默认 dataset-root/delta")
    for view in CAMERAS:
        parser.add_argument(f"--{view}-video-dir", type=Path, help=f"覆盖{view}视角视频目录")
    parser.add_argument("--reference-selection", type=Path, required=True)
    parser.add_argument("--reference-dataset", help="专家图来自其他数据集时指定其名称")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="+", help="例如 file-000 file-001；默认全部标注")
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--samples-per-episode", type=int, default=32)
    parser.add_argument("--terminal-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    result = build_reference_index(
        dataset_name=args.dataset_name or root.name,
        label_dir=args.label_dir or root / "labels", delta_dir=args.delta_dir or root / "delta",
        video_dirs={view: getattr(args, f"{view}_video_dir") or root / "videos" / name / "chunk-000"
                    for view, name in CAMERAS.items()},
        reference_selection=args.reference_selection, output_dir=args.output_dir,
        reference_dataset=args.reference_dataset, task=args.task, video_ids=args.video_ids,
        samples_per_episode=args.samples_per_episode, terminal_samples=args.terminal_samples, seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
