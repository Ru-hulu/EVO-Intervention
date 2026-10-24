# ruff: noqa: RUF002
"""固定流程：从左到右盖上三个块，再按红、绿、蓝的顺序掀开。

掀开时先掀哪个盖子取决于记住盖子下面的颜色，原始标注写的是"左/中/右盖子"，
步骤文字只写颜色，不泄漏位置。示范：每次盖上、掀开后的画面。
"""

from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec

SPEC = TaskSpec(
    kind=FIXED,
    steps=("Cover the left block with a lid.", "Cover the middle block with a lid.",
           "Cover the right block with a lid.", "Uncover the red block.", "Uncover the green block.",
           "Uncover the blue block."),
)
