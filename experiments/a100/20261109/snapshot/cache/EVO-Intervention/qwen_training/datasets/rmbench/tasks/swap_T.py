# ruff: noqa: RUF002
"""固定流程：双臂拿起两个 T 形块，回到初始姿态，再把两块分别放到对方的初始位姿。示范：四项各自完成时的画面。"""

from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec

SPEC = TaskSpec(
    rewind_safe=False,  # 尚未确认倒放与计数、阶段、进度标签回退的语义一致。
    kind=FIXED,
    steps=("Pick up both T-blocks with both arms.", "Move both arms back to their original position.",
           "Place the red T-block at the initial pose of the blue T-block.",
           "Place the blue T-block at the initial pose of the red T-block."),
)
