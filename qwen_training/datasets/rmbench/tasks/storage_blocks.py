# ruff: noqa: RUF002
"""次数由场景决定：每次拿起一到两块，放进篮子，直到桌上没有块。

拿起和放入的轮数由桌上块数决定，开局能看出。示范：拿起、放入各一次后的画面。
"""

from qwen_training.datasets.rmbench.spec import SCENE_COUNT, TaskSpec

SPEC = TaskSpec(
    kind=SCENE_COUNT,
    steps=("Pick up the next blocks.", "Place the held blocks into the basket."),
    step_prefixes=(("Pick up", 0), ("Place", 1)),
)
