#!/usr/bin/env python3
"""Sample 30 FPS MP4 videos at 5 FPS without moving their presentation times.

Output frame k comes from input frame 6*k, at time k/5 seconds (zero-based).
The last frame's display duration is shortened when necessary so the video
ends at exactly the original time. This may make the reported average FPS
slightly different from 5; consecutive frame timestamps are still 0.2 s apart.

Requires ffmpeg (libx264 and the setts bitstream filter) and ffprobe on PATH.
Only zero-start, constant-frame-rate inputs are accepted. Original files are
never modified. Existing outputs require --overwrite.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DEFAULT_ROOT = Path(
    "/Users/hongru/paper_project/RW-RL-Dataset/piperx_insert_copper_screw/"
    "videos/observation-recover_top/chunk-000/mainscope"
)


@dataclass
class VideoInfo:
    width: int
    height: int
    time_base: Fraction
    duration_ticks: int
    pts: list[int]
    packet_durations: list[int]

    @property
    def duration(self) -> Fraction:
        return self.duration_ticks * self.time_base


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_packets", "-show_entries",
        "stream=width,height,time_base,duration_ts,nb_frames:packet=pts,duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    if not data.get("streams") or not data.get("packets"):
        raise ValueError(f"No readable video stream: {path}")
    stream = data["streams"][0]
    packets = sorted(data["packets"], key=lambda packet: int(packet["pts"]))
    if int(stream["nb_frames"]) != len(packets):
        raise ValueError(f"Expected one video frame per packet: {path}")
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        time_base=Fraction(stream["time_base"]),
        duration_ticks=int(stream["duration_ts"]),
        pts=[int(packet["pts"]) for packet in packets],
        packet_durations=[int(packet["duration"]) for packet in packets],
    )


def validate_source(info: VideoInfo, source_fps: int, path: Path) -> None:
    frame_ticks = Fraction(1, source_fps) / info.time_base
    if info.time_base.numerator != 1 or frame_ticks.denominator != 1:
        raise ValueError(f"Time base cannot represent {source_fps} FPS exactly: {path}")
    expected_pts = [index * int(frame_ticks) for index in range(len(info.pts))]
    if info.pts != expected_pts:
        raise ValueError(f"Input must start at zero with constant {source_fps} FPS: {path}")
    if info.duration != Fraction(len(info.pts), source_fps):
        raise ValueError(f"Input duration disagrees with its frame timestamps: {path}")


def validate_output(source: VideoInfo, output: VideoInfo, stride: int) -> None:
    if (output.width, output.height) != (source.width, source.height):
        raise ValueError("Output resolution changed")
    expected_pts = source.pts[::stride]
    expected_times = [pts * source.time_base for pts in expected_pts]
    output_times = [pts * output.time_base for pts in output.pts]
    if output_times != expected_times:
        raise ValueError("Output frame count or presentation timestamps changed")
    if output.duration != source.duration:
        raise ValueError("Output duration differs from the original")
    expected_ends = expected_times[1:] + [source.duration]
    actual_ends = [(pts + duration) * output.time_base
                   for pts, duration in zip(output.pts, output.packet_durations)]
    if actual_ends != expected_ends:
        raise ValueError("Output frame display durations are incorrect")


def convert_video(
    source_path: Path,
    output_path: Path,
    info: VideoInfo,
    stride: int,
    crf: int,
    overwrite: bool,
) -> dict:
    selected_pts = info.pts[::stride]
    last_index = len(selected_pts) - 1
    frame_duration_ticks = stride * (info.pts[1] - info.pts[0]) if len(info.pts) > 1 else None
    if frame_duration_ticks is None:
        frame_duration_ticks = stride * info.duration_ticks
    last_duration_ticks = info.duration_ticks - selected_pts[-1]
    # Disable B-frames so packet index N is also the output presentation index.
    # setts changes only display durations, never the selected frame timestamps.
    # https://ffmpeg.org/ffmpeg-bitstream-filters.html#setts
    duration_filter = (
        f"setts=duration=if(eq(N\\,{last_index})\\,"
        f"{last_duration_ticks}\\,{frame_duration_ticks})"
    )
    descriptor, name = tempfile.mkstemp(prefix=f".{output_path.stem}.", suffix=".tmp.mp4", dir=output_path.parent)
    os.close(descriptor)
    temporary = Path(name)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-copyts", "-noautorotate", "-threads", "2", "-i", str(source_path),
        "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0", "-map_chapters", "0",
        "-vf", f"select=not(mod(n\\,{stride}))",
        "-fps_mode", "passthrough", "-enc_time_base", f"1:{info.time_base.denominator}",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-bf", "0", "-threads", "2",
        "-c:a", "copy", "-bsf:v", duration_filter,
        "-video_track_timescale", str(info.time_base.denominator),
        "-movie_timescale", str(info.time_base.denominator),
        "-movflags", "+faststart", str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        converted = probe_video(temporary)
        validate_output(info, converted, stride)
        if overwrite:
            temporary.replace(output_path)
        else:
            # Atomically refuse to replace a file created by another process.
            os.link(temporary, output_path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "file": source_path.name,
        "input_frames": len(info.pts),
        "output_frames": len(selected_pts),
        "duration_seconds": float(info.duration),
        "duration_exact": str(info.duration),
        "output_bytes": output_path.stat().st_size,
        "resolution": [info.width, info.height],
        "all_frame_timestamps_verified": True,
        "duration_unchanged": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT / "30FPS")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "5FPS")
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--target-fps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--crf", type=int, default=18, help="H.264 quality; 0 is lossless, default 18")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.source_fps <= 0 or args.target_fps <= 0 or args.source_fps % args.target_fps:
        parser.error("Source FPS must be a positive integer multiple of target FPS")
    if args.workers < 1 or not 0 <= args.crf <= 51:
        parser.error("--workers must be positive and --crf must be between 0 and 51")
    source_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    if source_dir == output_dir:
        parser.error("Input and output directories must differ")
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            parser.error(f"Required executable is missing: {tool}")
    if not source_dir.is_dir():
        parser.error(f"Input directory does not exist: {source_dir}")
    sources = sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mp4")
    if not sources:
        parser.error(f"No MP4 files found in {source_dir}")
    if not args.overwrite and any((output_dir / path.name).exists() for path in sources):
        parser.error("Output videos already exist; use --overwrite to replace them")

    stride = args.source_fps // args.target_fps
    infos = {}
    # Validate the whole input batch before creating any converted videos.
    for path in sources:
        info = probe_video(path)
        validate_source(info, args.source_fps, path)
        infos[path.name] = info
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input: {source_dir}\nOutput: {output_dir}", flush=True)
    print(f"{len(sources)} videos; retain frames 0, {stride}, {2 * stride}, ... at their original timestamps", flush=True)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(convert_video, path, output_dir / path.name, infos[path.name], stride, args.crf, args.overwrite): path
            for path in sources
        }
        for future in as_completed(jobs):
            path = jobs[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{len(results)}/{len(sources)}] {path.name}: "
                    f"{result['input_frames']} -> {result['output_frames']} frames; "
                    f"{result['duration_seconds']:.6f}s; timestamps and duration verified",
                    flush=True,
                )
            except Exception as error:
                message = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
                failures.append({"file": path.name, "error": message})
                print(f"FAILED {path.name}: {message}", file=sys.stderr, flush=True)

    results.sort(key=lambda result: result["file"])
    with (output_dir / "frame_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "output_frame_index", "source_frame_index", "timestamp_seconds", "duration_seconds"])
        for result in results:
            info = infos[result["file"]]
            selected = info.pts[::stride]
            for index, pts in enumerate(selected):
                end = selected[index + 1] if index + 1 < len(selected) else info.duration_ticks
                writer.writerow([
                    result["file"], index, index * stride,
                    f"{float(pts * info.time_base):.9f}", f"{float((end - pts) * info.time_base):.9f}",
                ])
    report = {
        "input_dir": str(source_dir), "output_dir": str(output_dir),
        "source_fps": args.source_fps, "target_fps": args.target_fps,
        "source_frame_stride": stride, "frame_indices_are_zero_based": True,
        "encoding": {"codec": "libx264", "crf": args.crf, "preset": "medium"},
        "last_frame_duration": "clipped to the original end time",
        "videos": results, "failures": failures,
    }
    (output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Finished: {len(results)} verified, {len(failures)} failed", flush=True)
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
