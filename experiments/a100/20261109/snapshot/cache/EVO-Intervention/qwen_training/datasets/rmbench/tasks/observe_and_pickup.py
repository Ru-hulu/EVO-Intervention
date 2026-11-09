# ruff: noqa: RUF002
"""固定流程：等挡板挡住架子上的目标物体，再从桌上拿起同样的物体。示范：两项各自完成时的画面。"""

from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec

SPEC = TaskSpec(
    rewind_safe=False,  # 尚未确认倒放与计数、阶段、进度标签回退的语义一致。
    kind=FIXED,
    steps=("Wait for the screen to cover the target object.",
           "Pick up the object that matches the target and lift it."),
)
