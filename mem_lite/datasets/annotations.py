"""读取进度标签使用的视频阶段标注。"""

import json
from pathlib import Path


def read_annotation(path):
    annotation = json.loads(Path(path).read_text(encoding="utf-8"))
    if annotation["frame_index_base"] != 0 or annotation["frame_range_convention"] != "[start_frame, end_frame)":
        raise ValueError("标注需采用 0 起始帧号和 [start_frame, end_frame) 区间")
    for stage in annotation["stages"]:
        if not 0 <= stage["start_frame"] < stage["end_frame"] <= annotation["total_frames"]:
            raise ValueError(f"阶段区间越界或为空：{stage}")
    return annotation
