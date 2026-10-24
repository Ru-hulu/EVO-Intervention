"""Render one RMBench episode with three cameras and annotation-derived stages.

Example (on 242, with the RoboDojo simulation Python):
  python -m qwen_training.datasets.rmbench.visualize \
    --root /home/zhaobo/datasets/rmbench/data --task press_button --episode 0 \
    --source-fps 16.6666666667 --robodojo-root /home/zhaobo/robodojo-benchmark/RoboDojo \
    --out outputs/rmbench/visualization/videos/press_button_0.mp4

Requires numpy, h5py, opencv-python, the XPolicyLab image decoder, and ffmpeg.
The source FPS is explicit because these HDF5 files have no timestamps. Stages
and progress come from the existing label converter, not model predictions.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import cv2
import h5py
import numpy as np

from qwen_training.datasets.rmbench.labels import CAMERAS, episode_labels, sample_frames
from qwen_training.datasets.rmbench.tasks import TASKS


WIDTH, HEIGHT = 1440, 1240
WHITE, MUTED = (235, 235, 235), (180, 180, 180)
BLUE = (235, 180, 65)
COLORS = [(220, 160, 65), (100, 195, 100), (90, 120, 235), (205, 125, 205),
          (65, 205, 220), (200, 190, 100), (160, 100, 235), (110, 190, 245),
          (200, 130, 100), (160, 210, 170), (160, 160, 230)]


def text(image, value, position, color=WHITE, scale=0.65):
    cv2.putText(image, str(value), position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def wrapped(image, value, x, y, width=110, lines=3, color=WHITE):
    for line in textwrap.wrap(value, width=width, max_lines=lines, placeholder="..."):
        text(image, line, (x, y), color)
        y += 25


def timeline(image, slots, steps, index):
    """Whole-episode frame axis; repeated instances share their task's color."""
    total = slots[-1]["end"] + 1
    left, right, top, bottom = 35, WIDTH - 35, 725, 785
    position = lambda frame: left + round((right - left) * frame / total)
    text(image, "FULL TRAJECTORY | colors = tasks; numbered blocks = instances", (20, 690))
    for number, slot in enumerate(slots):
        x0, x1 = position(slot["start"]), position(slot["end"] + 1)
        color = COLORS[slot["step"] % len(COLORS)]
        cv2.rectangle(image, (x0, top), (x1, bottom), color, -1)
        active = slot["start"] <= index <= slot["end"]
        cv2.rectangle(image, (x0, top), (x1, bottom), WHITE if active else (30, 30, 30), 3 if active else 1)
        text(image, number + 1, ((x0 + x1) // 2 - 8, top + 38), (15, 15, 15), scale=0.7)
        # Exact transition frames, staggered to keep nearby boundaries readable.
        y = 713 if number % 2 == 0 else 701
        text(image, slot["start"], (x0, y), MUTED, scale=0.44)
    text(image, total - 1, (right - 40, 713), MUTED, scale=0.44)
    cv2.line(image, (left, 815), (right, 815), MUTED, 1)
    for value in np.linspace(0, total - 1, 6, dtype=int):
        x = position(int(value))
        cv2.line(image, (x, 810), (x, 821), MUTED, 1)
        text(image, value, (max(left, x - 17), 842), MUTED, scale=0.5)
    pointer = position(index)
    cv2.line(image, (pointer, top - 8), (pointer, bottom + 17), (75, 75, 255), 3)
    text(image, "Source frame (0-based); legend ranges include both endpoints", (20, 873), MUTED, scale=0.55)
    for number, slot in enumerate(slots):
        column, row = divmod(number, 8)
        x, y = 20 + column * 710, 910 + row * 38
        color = COLORS[slot["step"] % len(COLORS)]
        cv2.rectangle(image, (x, y - 13), (x + 12, y), color, -1)
        value = (f"#{number + 1} [{slot['start']}..{slot['end']}] "
                 f"{steps[slot['step']]} (#{slot['instance']})")
        for line_number, line in enumerate(textwrap.wrap(value, width=75, max_lines=2, placeholder="...")):
            text(image, line, (x + 20, y + line_number * 16), color, scale=0.46)


def render_frame(h5, decode, labels, record, raw_segment, task, episode, instruction, fps):
    image = np.full((HEIGHT, WIDTH, 3), (24, 24, 24), np.uint8)
    index = record["index"]
    slots, steps = labels["trace"], labels["steps"]
    current_index = next(i for i, slot in enumerate(slots) if slot["start"] <= index <= slot["end"])
    slot = slots[current_index]
    text(image, f"RMBench | {task} | episode {episode}", (20, 35), scale=0.9)
    text(image, f"Frame {index}/{slots[-1]['end']} | playback time {index / fps:.2f}s | playback FPS {fps:g}",
         (20, 67), MUTED)
    wrapped(image, "Instruction: " + instruction, 20, 100, width=115, lines=3)

    for view, camera in enumerate(CAMERAS):
        rgb = decode(h5[f"observation/{camera}/rgb"][index])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Letterbox instead of stretching images from another camera resolution.
        height, width = bgr.shape[:2]
        ratio = min(480 / width, 360 / height)
        resized = cv2.resize(bgr, (round(width * ratio), round(height * ratio)))
        y = 180 + (360 - resized.shape[0]) // 2
        x = view * 480 + (480 - resized.shape[1]) // 2
        image[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        cv2.rectangle(image, (view * 480, 180), (view * 480 + 479, 212), (25, 25, 25), -1)
        text(image, camera, (view * 480 + 12, 203))

    text(image, f"CURRENT STAGE {record['stage'] + 1}/{len(steps)}"
         f" | instance {slot['instance']}/{labels['step_counts'][slot['step']]}"
         f" | interval {current_index + 1}/{len(slots)}", (20, 577), BLUE, scale=0.8)
    wrapped(image, steps[slot["step"]], 20, 610, width=110, lines=2, color=BLUE)
    text(image, f"Raw action segment {raw_segment + 1} | frames {slot['start']}..{slot['end']}"
         f" | completed {sum(record['counts'])}/{len(slots)}",
         (20, 666), MUTED)
    timeline(image, slots, steps, index)
    text(image, "Annotation-derived stages; all source frames are retained at default output FPS.",
         (20, 1220), MUTED, scale=0.5)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="RMBench data directory")
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--source-fps", type=float, required=True)
    parser.add_argument("--output-fps", type=float, help="Default: preserve source FPS")
    parser.add_argument("--robodojo-root", type=Path, help="Directory containing XPolicyLab/")
    parser.add_argument("--out", type=Path, required=True, help="Output .mp4 file")
    args = parser.parse_args()
    output_fps = args.output_fps or args.source_fps
    if not 0 < output_fps <= args.source_fps:
        parser.error("Require 0 < output FPS <= source FPS")
    if args.out.suffix.lower() != ".mp4":
        parser.error("--out must end in .mp4")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required")
    if args.robodojo_root:
        sys.path.insert(0, str(args.robodojo_root))
    from XPolicyLab.utils.process_data import decode_image_bit

    demo = args.root / args.task / "demo_clean"
    annotation = json.loads((demo / "language_annotation.json").read_text())
    segments = annotation[f"episode_{args.episode}"]
    labels = episode_labels(segments, TASKS[args.task], args.source_fps, sample_fps=args.source_fps)
    instruction = json.loads((demo / "instructions" / f"episode{args.episode}.json").read_text())["seen"][0]
    total = sum(length for _, length in segments)
    raw_ends = np.cumsum([length for _, length in segments])
    source = demo / "data" / f"episode{args.episode}.hdf5"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as h5:
        counts = [len(h5[f"observation/{camera}/rgb"]) for camera in CAMERAS]
        if any(count != total for count in counts):
            raise ValueError(f"Annotation frames {total} != camera frames {counts}")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
                   "-pixel_format", "bgr24", "-video_size", f"{WIDTH}x{HEIGHT}",
                   "-framerate", str(output_fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   str(args.out)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            indices = sample_frames(total, args.source_fps, output_fps)
            for index in indices:
                raw = int(np.searchsorted(raw_ends, index, side="right"))
                frame = render_frame(h5, decode_image_bit, labels, labels["frames"][index], raw,
                                     args.task, args.episode, instruction, args.source_fps)
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"ffmpeg exited with {returncode}")
    print(f"Wrote {len(indices)} frames: {args.out}")


if __name__ == "__main__":
    main()
