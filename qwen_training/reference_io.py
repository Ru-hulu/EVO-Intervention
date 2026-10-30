"""五张专家图、六个观测时刻（其中一个三视角）→ 六行 progress/done。"""

import re

FORMAT_VERSION = "reference_window_v2"
IMAGE_COUNT = 13
FRAME_COUNT = 6
NA_ROW = "progress=NA | done=NA"
OUTPUT_INSTRUCTION = (
    "\n按输入frame顺序输出6行，不按时间重排。"
    "progress为整项任务进度，0～1，三位小数；done为已完成1、未完成0。"
    "空白帧两项均为NA。每行：progress=数值 | done=整数。"
    "不添加其他内容。"
)
ROW_PATTERN = re.compile(r"progress=([01]\.\d{3})\s*\|\s*done=([01])")


def frame_views(frame):
    return ("top", "left", "right") if frame.get("is_current", False) else ("top",)


def make_reference_message(task, frames=None):
    """占位符按帧组展开；三视角可在任意位置，不提供时间/进度标签。"""
    if frames is None:
        frames = [{"valid": True, "is_current": i == 5} for i in range(FRAME_COUNT)]
    if len(frames) != FRAME_COUNT or sum(bool(f.get("is_current")) for f in frames) != 1:
        raise ValueError("需要六个时刻，其中恰有一个当前时刻的三视角")
    content = [{"type": "text", "text": f"任务：{task}\n成功专家轨迹参考，按时间排列："}]
    for i in range(1, 6):
        content.extend([
            {"type": "text", "text": f"\nreference-{i:03d}："}, {"type": "image"},
        ])
    content.append({"type": "text", "text": "\n观测帧："})
    names = {"top": "顶部", "left": "左腕", "right": "右腕"}
    for i, frame in enumerate(frames, 1):
        note = "（空白）" if not frame["valid"] else "（当前时刻）" if frame.get("is_current") else ""
        content.append({"type": "text", "text": f"\nframe-{i:03d}{note}："})
        for view in frame_views(frame):
            if frame.get("is_current"):
                content.append({"type": "text", "text": f"{names[view]}："})
            content.append({"type": "image"})
    content.append({"type": "text", "text": OUTPUT_INSTRUCTION})
    return [{"role": "user", "content": content}]


def format_frame_targets(targets):
    if len(targets) != FRAME_COUNT:
        raise ValueError("每个窗口需要六组 progress/done 标签")
    rows = []
    for target in targets:
        if not target.get("valid", True):
            rows.append(NA_ROW)
            continue
        progress, done = target["progress"], target["done"]
        if not (0 <= progress <= 1 and done in (0, 1)):
            raise ValueError("有效帧的 progress/done 超出约定范围")
        rows.append(f"progress={progress:.3f} | done={int(done)}")
    return "\n".join(rows)


def parse_frame_predictions(response):
    """合法 NA 行与无法解析的行分开；数值评价只用 valid=True 的行。"""
    lines = response.strip().splitlines()
    if len(lines) != FRAME_COUNT:
        return [None] * FRAME_COUNT
    result = []
    for line in lines:
        line = line.strip()
        if line == NA_ROW:
            result.append(dict(progress=None, done=None, valid=False))
            continue
        match = ROW_PATTERN.fullmatch(line)
        if match is None or float(match[1]) > 1:
            result.append(None)
        else:
            result.append(dict(progress=float(match[1]), done=int(match[2]), valid=True))
    return result
