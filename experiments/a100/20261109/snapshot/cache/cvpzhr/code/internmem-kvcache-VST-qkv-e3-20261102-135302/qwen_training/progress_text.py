"""进度作为答案末尾的普通文本生成；数值解析仅用于推理与评估。"""

import re
import unicodedata


PROGRESS_INSTRUCTION = "最后一行按“progress：数值”的格式输出目标任务进度，数值范围 0～1，保留三位小数。"


def append_progress(response, progress):
    """无目标任务时不追加进度；其余样本与状态文本一起做 token CE。"""
    return response if progress is None else f"{response}\nprogress：{progress:.3f}"


def parse_progress(response):
    """只读取末尾唯一的 progress 字段；缺失、截断、越界返回 None。"""
    text = unicodedata.normalize("NFKC", response)
    fields = re.findall(r"\bprogress\s*:", text, flags=re.IGNORECASE)
    match = re.search(r"\bprogress\s*:\s*([01]\.\d{3})\s*$", text, flags=re.IGNORECASE)
    if len(fields) != 1 or match is None:
        return None
    value = float(match.group(1))
    return value if 0 <= value <= 1 else None
