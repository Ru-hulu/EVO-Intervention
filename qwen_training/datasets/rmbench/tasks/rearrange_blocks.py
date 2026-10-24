# ruff: noqa: RUF002
"""固定流程：把两垫之间的块移到空垫上，按按钮，再把另一块移到两垫之间。

原始标注写明了左右，步骤文字不写。示范：三项各自完成时的画面。
"""

from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec

SPEC = TaskSpec(
    kind=FIXED,
    steps=("Move the block between the two mats onto the empty mat.", "Press the button.",
           "Move the other block to the space between the two mats."),
)
