# ruff: noqa: RUF002, RUF003
"""每个 RMBench 任务的标注规则与示范设计（proposal_progress 3.2.1）。"""

from dataclasses import dataclass

FIXED = "fixed"              # 固定流程：每条 episode 的步骤和次数相同，实例按顺序对应各项。
SCENE_COUNT = "scene_count"  # 次数由场景决定：开局能从画面看出每项做几次。
SEARCH = "search"            # 试错：做几次取决于何时成功，开局不可知。


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    # 输入给模型的通用步骤文字：不含只有记忆才知道的答案，不写次数。
    steps: tuple[str, ...]
    # 非固定流程：原始标注文字的前缀 → 项下标。固定流程按实例顺序对应，不需要。
    step_prefixes: tuple[tuple[str, int], ...] = ()
    # 一个实例包含的段数；None 表示一组连续的相同文字段为一个实例。
    segments_per_instance: int | None = None
    # 这些标注文字的段并入前一个实例（原始标注把同一次动作的收尾标成了别的文字）。
    merge_into_previous: tuple[str, ...] = ()
