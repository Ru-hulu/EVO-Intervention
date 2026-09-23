"""Synchronize original LeRobot camera videos with existing LLMD/ACC scores.

Requires ffmpeg with libass/libx264, Pillow, matplotlib, numpy and pyarrow.
Scores use causal sample-and-hold, never interpolation or new inference.
"""

# ruff: noqa: RUF001 - Chinese punctuation in the video captions is intentional.

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib as mpl
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pyarrow.parquet as pq

mpl.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

WIDTH, HEIGHT = 1920, 1080
BG = "#101827"
PANEL = "#182438"
TEXT = "#eef3fb"
MUTED = "#a7b5c9"
COLORS = ["#75b7ff", "#54d5ae", "#ffb76b"]


def load_inputs(report, dataset=None):
    summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    with (report / "scores.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2 or any(row["episode_id"] != summary["episode_id"] for row in rows):
        raise ValueError("Scores must contain at least two rows matching the report episode")
    provenance = summary["provenance"]
    root = Path(dataset or provenance["dataset_root"])
    info = json.loads((root / "meta/info.json").read_text())
    matches = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        columns = [name for name in pq.read_schema(path).names if not name.startswith("stats/")]
        matches.extend(
            pq.read_table(
                path, columns=columns, filters=[("episode_index", "=", provenance["episode_index"])]
            ).to_pylist()
        )
    if len(matches) != 1:
        raise ValueError("Expected exactly one episode metadata row")
    meta = matches[0]
    fps = float(info["fps"])
    times = np.array([float(row["time_s"]) for row in rows])
    frames = np.array([int(row["frame_index"]) for row in rows])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Score times must be finite and strictly increasing")
    if not np.allclose(times, frames / fps, atol=0.001, rtol=0):
        raise ValueError("Score timestamps do not match source frame indices")
    count = int(meta["length"])
    if frames[-1] >= count or times[0] != 0 or count / fps - times[-1] > 1.5 * np.median(np.diff(times)):
        raise ValueError("Scores must cover the original episode; do not hold a stale score over missing sections")
    cameras = []
    for name in ("base", "left_wrist", "right_wrist"):
        key = provenance["camera_map"][name]
        prefix = f"videos/{key}/"
        source = root / info["video_path"].format(
            video_key=key, chunk_index=meta[prefix + "chunk_index"], file_index=meta[prefix + "file_index"]
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        offset = float(meta[prefix + "from_timestamp"])
        if float(meta[prefix + "to_timestamp"]) - offset < count / fps - 1 / fps:
            raise ValueError(f"Camera {name} is shorter than the episode")
        cameras.append({"camera": name, "path": str(source), "offset_s": offset})
    return summary, rows, times, fps, count, cameras


def background_assets(folder, summary, rows, times, duration, font_path, fps, axis_limits=None, title=None):
    acc_only = summary.get("metric_mode") == "acc"
    background = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(background)

    def label(xy, text, size=24, fill=TEXT):
        draw.text(xy, text, font=ImageFont.truetype(str(font_path), size), fill=fill)

    name = summary["episode_id"].rsplit("/", 1)[-1]
    metric_title = "ACC" if acc_only else "LLMD / ACC"
    label((24, 15), f"{title or name}  ·  {metric_title} 同步回放", 35)
    model = Path(summary["feature_metadata"]["checkpoint"]).name
    outcome = summary["provenance"].get("label", "unknown")
    outcome = {"failure": "失败", "success": "成功", "demonstration": "训练示范（无原生结果标签）"}.get(outcome, "未知")
    label((25, 67), f"评估模型：{model}  |  记录结果：{outcome}", 21, MUTED)
    for box in [(24, 134, 984, 674), (24, 720, 494, 990), (514, 720, 984, 990)]:
        draw.rounded_rectangle(box, radius=12, fill=PANEL)
    label((24, 100), "环境相机", 23)
    label((24, 684), "左腕相机", 23)
    label((514, 684), "右腕相机", 23)
    for x, card_title, color in zip(
        [1020, 1314, 1608],
        ["双臂最大 ACC" if acc_only else "LLMD 平方距离", "左臂 ACC · EMA", "右臂 ACC · EMA"],
        COLORS,
        strict=True,
    ):
        draw.rounded_rectangle((x, 119, x + 276, 232), radius=12, fill=PANEL)
        label((x + 15, 126), card_title, 23, color)
    query_hz = 1 / summary["query_interval_s"]
    label((24, 1023), f"原始视频 {fps:g} FPS · ACC {query_hz:g} Hz · H={summary['action_horizon']} / R={summary['action_horizon'] - summary['overlap_steps']} · 原速播放", 22, MUTED)
    threshold_note = (
        "阈值未校准"
        if summary["acc_threshold"] is None or (not acc_only and summary["llmd_threshold"] is None)
        else "已有校准阈值"
    )
    label((1020, 985), f"离线回放 · {threshold_note} · 高分不等于已确认失败", 23, "#f5cf83")
    scale_note = "对比视频纵轴一致" if axis_limits else "本视频自动纵轴"
    footer = (
        f"ACC：α={summary['ema_alpha']:g}，v_min={summary['v_min_m']:g} m"
        if acc_only
        else f"LLMD 参考：{summary['reference_samples']} 个观测"
    )
    label((1020, 1023), f"{footer}；{scale_note}", 21, MUTED)

    chart_width, chart_height = 900, 685
    chart_x, chart_y = 1008, 289
    fig = Figure(figsize=(chart_width / 100, chart_height / 100), dpi=100, facecolor=BG)
    canvas = FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.078, top=0.94, hspace=0.48)
    series = [
        [("acc_raw_left", COLORS[1], "Left"), ("acc_raw_right", COLORS[2], "Right")]
        if acc_only
        else [("llmd", COLORS[0], "LLMD")],
        [("acc_left", COLORS[1], "Left"), ("acc_right", COLORS[2], "Right")],
        [("mae_mm_left", COLORS[1], "Left"), ("mae_mm_right", COLORS[2], "Right")],
    ]
    for ax, plot_title, fields in zip(
        axes,
        [
            "ACC: raw inconsistency" if acc_only else "LLMD: max token squared distance",
            "ACC: smoothed inconsistency",
            "Overlap XYZ error (mm)",
        ],
        series,
        strict=True,
    ):
        ax.set_facecolor(BG)
        for field, color, legend in fields:
            values = [float(row[field]) if row[field] else np.nan for row in rows]
            ax.plot(times, values, color=color, linewidth=1.5, label=legend)
        ax.set_title(plot_title, loc="left", color=TEXT, fontsize=12, pad=10)
        ax.set_xlim(0, duration)
        ax.set_ylim(bottom=0)
        ax.grid(color="#304259", alpha=0.5)
        ax.tick_params(colors=MUTED, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#34465c")
        if len(fields) > 1:
            ax.legend(loc="upper left", frameon=False, labelcolor=TEXT, ncol=2, fontsize=10)
    axes[-1].set_xlabel("Recorded episode time (s)", color=MUTED, fontsize=11)
    canvas.draw()
    if axis_limits:
        for ax, key in zip(axes, ["acc_raw" if acc_only else "llmd", "acc", "mae_mm"], strict=True):
            ax.set_ylim(0, axis_limits[key])
        canvas.draw()
    background.paste(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB"), (chart_x, chart_y))
    background.save(folder / "background.png")
    cursor = Image.new("RGBA", (4, chart_height), (0, 0, 0, 0))
    pen = ImageDraw.Draw(cursor)
    for ax in axes:
        rect = ax.get_window_extent()
        pen.rectangle((0, chart_height - rect.y1, 3, chart_height - rect.y0), fill=(255, 235, 153, 235))
    cursor.save(folder / "cursor.png")
    bounds = axes[0].get_window_extent()
    return chart_x + bounds.x0, bounds.width, chart_y


def ass_time(seconds):
    ticks = round(seconds * 100)
    return f"{ticks // 360000}:{ticks // 6000 % 60:02}:{ticks // 100 % 60:02}.{ticks % 100:02}"


def subtitle_assets(folder, rows, times, duration, font_family, *, acc_only=False):
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 2",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_family},26,&H00FBF3EE,&H00FFFFFF,&H00271810,&H00271810,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def event(start, end, x, y, text, size=44, color="FBF3EE"):
        if end > start:
            style = f"{{\\an7\\pos({x},{y})\\fs{size}\\c&H{color}&}}"
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{style}{text}")

    for index, row in enumerate(rows):
        start = times[index]
        if start >= duration:
            break
        end = min(duration, times[index + 1] if index + 1 < len(times) else duration)
        key = "acc_max" if acc_only else "llmd"
        number = (f"{float(row[key]):.3f}" if acc_only else f"{float(row[key]):,.0f}") if row[key] else "--"
        event(start, end, 1035, 163, number, color="FFB775")
        for x, key, color in [(1329, "acc_left", "AED554"), (1623, "acc_right", "6BB7FF")]:
            event(start, end, x, 163, f"{float(row[key]):.3f}" if row[key] else "--", color=color)
        event(
            start,
            end,
            1020,
            245,
            f"最近评分：{start:.1f} s   |   原始帧：{row['frame_index']}   |   首次 ACC 需两次预测",
            21,
        )
    for tick in range(int(np.ceil(duration * 10))):
        stamp = tick / 10
        event(stamp, min(stamp + 0.1, duration), 1530, 20, f"视频时间  {int(stamp // 60):02}:{stamp % 60:04.1f}", 29)
    (folder / "values.ass").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(
    report,
    output,
    *,
    dataset=None,
    font_path=None,
    font_family="Noto Sans CJK SC",
    limit_seconds=None,
    axis_limits_path=None,
    title=None,
):
    report, output = Path(report).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise FileNotFoundError(f"{binary} must be on PATH")
    font_path = Path(font_path or "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font_path.is_file():
        raise FileNotFoundError("Pass --font pointing to a local Chinese font")
    summary, rows, times, fps, source_count, cameras = load_inputs(report, dataset)
    acc_only = summary.get("metric_mode") == "acc"
    axis_limits = None
    if axis_limits_path:
        axis_limits = json.loads(Path(axis_limits_path).read_text(encoding="utf-8"))
        for key, fields in {
            **({"acc_raw": ["acc_raw_left", "acc_raw_right"]} if acc_only else {"llmd": ["llmd"]}),
            "acc": ["acc_left", "acc_right"],
            "mae_mm": ["mae_mm_left", "mae_mm_right"],
        }.items():
            maximum = float(axis_limits[key])
            values = [float(row[field]) for row in rows for field in fields if row[field]]
            if not np.isfinite(maximum) or maximum <= 0 or max(values, default=0) > maximum:
                raise ValueError(f"Invalid shared axis limit or clipped data: {key}")
    count = source_count if limit_seconds is None else min(source_count, round(limit_seconds * fps))
    if count < 1:
        raise ValueError("Video must contain at least one frame")
    duration = count / fps
    output.parent.mkdir(parents=True, exist_ok=True)
    folder = output.parent / (output.stem + "_assets")
    folder.mkdir(exist_ok=True)
    x0, plot_width, chart_y = background_assets(
        folder, summary, rows, times, source_count / fps, font_path, fps, axis_limits, title
    )
    subtitle_assets(folder, rows, times, duration, font_family, acc_only=acc_only)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-n",
        "-filter_complex_threads",
        "2",
        "-loop",
        "1",
        "-framerate",
        str(fps),
        "-i",
        "background.png",
    ]
    for camera in cameras:
        command += ["-threads", "2", "-ss", str(camera["offset_s"]), "-i", camera["path"]]
    command += ["-loop", "1", "-framerate", str(fps), "-i", "cursor.png"]
    filters = []
    for index, (width, height, x, y) in enumerate([(960, 540, 24, 134), (470, 270, 24, 720), (470, 270, 514, 720)], 1):
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,fps={fps},scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x182438,setsar=1[cam{index}]"
        )
        previous = "0:v" if index == 1 else f"panel{index-1}"
        filters.append(f"[{previous}][cam{index}]overlay=x={x}:y={y}:shortest=1[panel{index}]")
    filters += [
        f"[panel3][4:v]overlay=x='{x0:.5f}-2+{plot_width:.5f}*t/{source_count / fps:.8f}':y={chart_y}:shortest=1[timeline]",
        "[timeline]subtitles=values.ass:fontsdir=.[out]",
    ]
    graph = ";\n".join(filters)
    (folder / "filter.txt").write_text(graph, encoding="utf-8")
    command += [
        "-filter_complex_script",
        "filter.txt",
        "-map",
        "[out]",
        "-an",
        "-frames:v",
        str(count),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output),
    ]
    (folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    print(f"Rendering {source_count} source frames; output={output}", flush=True)
    with (folder / "ffmpeg.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=folder, stderr=log, check=True)
    probe = json.loads(
        subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(output)])
    )
    stream = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    if int(stream["nb_frames"]) != count or abs(float(stream["duration"]) - duration) > 1 / fps:
        raise RuntimeError("Encoded video duration/frame count does not match source alignment")
    manifest = {
        "report": str(report),
        "output": str(output),
        "source_cameras": cameras,
        "frames": count,
        "fps": fps,
        "duration_s": duration,
        "score_samples": len(rows),
        "score_display": "causal sample-and-hold; no interpolation",
        "model": summary["feature_metadata"]["checkpoint"],
        "source_provenance": summary["provenance"],
        "axis_limits": axis_limits,
        "display_title": title,
        "ffprobe": probe,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"VERIFIED {count} frames, {duration:.3f}s: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Directory containing scores.csv and summary.json")
    parser.add_argument("--output", required=True, help="New MP4 output path")
    parser.add_argument("--dataset", help="Optional source dataset location override")
    parser.add_argument("--font", help="Path to a Chinese font; Linux Noto CJK is the default")
    parser.add_argument("--font-family", default="Noto Sans CJK SC")
    parser.add_argument("--limit-seconds", type=float, help="Short rendering check; omit for the complete episode")
    parser.add_argument("--axis-limits", help="JSON with llmd/acc/mae_mm upper bounds shared across comparison videos")
    parser.add_argument("--title", help="Display label; does not change the source episode identity")
    args = parser.parse_args()
    render(
        args.report,
        args.output,
        dataset=args.dataset,
        font_path=args.font,
        font_family=args.font_family,
        limit_seconds=args.limit_seconds,
        axis_limits_path=args.axis_limits,
        title=args.title,
    )


if __name__ == "__main__":
    main()
