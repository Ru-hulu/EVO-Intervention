"""按观测时刻读取图像；三视角使用相同的原视频相对时间戳。"""

from pathlib import Path

from qwen_training.reference_io import FRAME_COUNT, frame_views


def decode_video_frames(path, times):
    """只解码所需区间，按各相机实际帧率取最近帧，每次读取后关闭解码器。"""
    import av

    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = stream.average_rate
        start_pts = stream.start_time or 0
        wanted = {time: round(time * fps) for time in times}
        indices = set(wanted.values())
        timestamp = start_pts + int(min(indices) / fps / stream.time_base)
        video.seek(timestamp, stream=stream, backward=True)
        decoded = {}
        iterator = video.decode(stream)
        try:
            for frame in iterator:
                index = round((frame.pts - start_pts) * stream.time_base * fps)
                if index in indices:
                    decoded[index] = frame.to_image().convert("RGB")
                if index >= max(indices):
                    break
        finally:
            iterator.close()
            # PyAV 12 提前停止解码时显式关闭 codec，避免数据 worker 累积解码线程。
            if hasattr(stream.codec_context, "close"):
                stream.codec_context.close(strict=False)
    missing = indices - decoded.keys()
    if missing:
        raise ValueError(f"视频缺少对应时间戳的帧：{path}, {sorted(missing)}")
    return {time: decoded[index] for time, index in wanted.items()}


def load_window_images(sample, base_dir):
    """支持视频帧索引或 frames[].images 文件路径，按帧组展开为13张 RGB 图。"""
    from PIL import Image

    base_dir = Path(base_dir)
    frames = sample["frames"]
    if len(sample["reference_images"]) != 5 or len(frames) != FRAME_COUNT:
        raise ValueError("窗口需要五张专家图和六个观测时刻")
    if sum(bool(f.get("is_current")) for f in frames) != 1:
        raise ValueError("六个时刻中恰有一个当前三视角时刻")

    def read_image(path):
        with Image.open(base_dir / path) as image:
            return image.convert("RGB")

    images = [read_image(path) for path in sample["reference_images"]]
    groups = []
    if "video_paths" in sample:
        decoded = {}
        for view in ("top", "left", "right"):
            times = {frame["time_seconds"] for frame in frames
                     if frame["valid"] and view in frame_views(frame)}
            decoded[view] = decode_video_frames(base_dir / sample["video_paths"][view], times)
        for frame in frames:
            groups.append([decoded[view][frame["time_seconds"]] for view in frame_views(frame)]
                          if frame["valid"] else None)
    else:
        for frame in frames:
            groups.append([read_image(frame["images"][view]) for view in frame_views(frame)]
                          if frame["valid"] else None)
    # 空白帧采用该窗口真实主视角的尺寸，不读取前一个 episode，也不复制首帧。
    size = next(group[0].size for group in groups if group is not None)
    for group in groups:
        images.extend(group if group is not None else [Image.new("RGB", size, (0, 0, 0))])
    return images
