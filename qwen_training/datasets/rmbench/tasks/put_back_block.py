# ruff: noqa: RUF002
"""固定流程：把块移到中间，按按钮，再放回原来的垫子。

原始标注写明了原来是哪个垫子（front/back/left/right），这正是要记住的答案，步骤文字不写。
示范：三项各自完成时的画面。
"""

from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec

SPEC = TaskSpec(
    kind=FIXED,
    steps=("Move the block to the center.", "Press the button.", "Put the block back on its original mat."),
)
