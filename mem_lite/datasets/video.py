"""按真实时间戳取 10 FPS 候选帧，过滤静止观测并保留标注边界。"""

from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageFilter


def motion_fraction(image, reference, width, pixel_delta):
    """返回低分辨率灰度图及明显变化的像素比例；保留局部动作，不取全图平均亮度差。"""
    height = max(1, round(image.height * width / image.width))
    gray = np.asarray(
        image.convert("L").resize((width, height), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(1)),
        dtype=np.int16,
    )
    changed = None if reference is None else float(np.mean(np.abs(gray - reference) > pixel_delta))
    return gray, changed


def iter_timed_frames(video_path, sample_fps):
    """每个原帧返回 (帧号, 实际秒数, 是否为固定时钟候选, AV帧)。

    每个采样时刻选择当时最新的原视频帧，不读取未来图像。
    用一帧前瞻确定当前帧覆盖哪些采样时刻；低帧率源不会重复入队。
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps 必须大于 0")
    with av.open(str(video_path)) as video:
        stream = video.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate)
        previous, origin, tick = None, None, 0
        for index, frame in enumerate(video.decode(stream)):
            raw_time = float(frame.pts * stream.time_base) if frame.pts is not None else index / fps
            if origin is None:
                origin = raw_time
            time = raw_time - origin
            if previous is not None:
                if time <= previous[1]:
                    raise ValueError(f"视频时间戳未递增：frame={index}")
                candidate = False
                while tick / sample_fps < time - 1e-9:
                    candidate = True
                    tick += 1
                yield (*previous[:2], candidate, previous[2])
            previous = (index, time, frame)
        if previous is not None:
            duration = (
                float(stream.duration * stream.time_base) if stream.duration is not None else previous[1] + 1 / fps
            )
            candidate = tick / sample_fps < duration - 1e-9
            yield (*previous[:2], candidate, previous[2])


def extract_observations(video_path, boundary_frames, frames_dir, config):
    """扫描整条主视角视频。强制边界参与队列，但不重置固定候选时钟。"""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    with av.open(str(video_path)) as video:
        stream = video.streams.video[0]
        info = {"fps": float(stream.average_rate), "width": stream.width, "height": stream.height}
    observations, timestamps, kept_frames = [], [], []
    reference = None
    for index, time, candidate, frame in iter_timed_frames(video_path, config["sample_fps"]):
        timestamps.append(time)
        boundary = index in boundary_frames
        if not candidate and not boundary and index != 0:
            continue
        image = frame.to_image()
        gray, changed = motion_fraction(image, reference, config["difference_width"], config["pixel_delta"])
        keep = boundary or reference is None or changed >= config["min_changed_fraction"]
        observations.append(
            {
                "frame_index": index,
                "time_seconds": time,
                "candidate_10fps": candidate,
                "annotation_boundary": boundary,
                "changed_fraction": changed,
                "kept": keep,
            }
        )
        if keep:
            image.save(frames_dir / f"{index:06d}.jpg", quality=92)
            reference = gray  # 只更新到上一张实际保留的观测。
            kept_frames.append(index)
    info.update(frame_count=len(timestamps), timestamps=timestamps, observations=observations, kept_frames=kept_frames)
    return info


def save_missing_frames(video_path, indices, frames_dir):
    """补出各训练样本自己的扰动目标边界；不把这些帧写入公共观测队列。"""
    frames_dir = Path(frames_dir)
    missing = {i for i in indices if not (frames_dir / f"{i:06d}.jpg").exists()}
    if not missing:
        return
    with av.open(str(video_path)) as video:
        stream = video.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(video.decode(stream)):
            if index in missing:
                frame.to_image().save(frames_dir / f"{index:06d}.jpg", quality=92)
                missing.remove(index)
            if not missing:
                break
    if missing:
        raise ValueError(f"视频缺少目标边界帧：{sorted(missing)}")
