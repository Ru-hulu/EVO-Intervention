# ruff: noqa: RUF002
"""次数由场景决定：左键、中键各按桌上数字牌对应的次数，最后按右键确认。

每段是一次按压。次数只能从首帧的数字牌读出，步骤文字不写次数。
示范：每种按键各一次按压后的画面，不展示示范自己的次数。
"""

from qwen_training.datasets.rmbench.spec import SCENE_COUNT, TaskSpec

SPEC = TaskSpec(
    kind=SCENE_COUNT,
    steps=("Press the left button once.", "Press the middle button once.", "Press the right button to confirm."),
    step_prefixes=(("Press the left", 0), ("Press the middle", 1), ("Press the confirm", 2)),
    segments_per_instance=1,
)
