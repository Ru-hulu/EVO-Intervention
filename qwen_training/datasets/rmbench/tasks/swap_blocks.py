# ruff: noqa: RUF002
"""固定流程：借助空托盘移动三次交换两块，最后按按钮。

三次移动从哪一块开始每条不同，按"第几次移动"对应。原始标注把"左块移到右托盘"的最后一段
单独标成了"右块移到右托盘"，并入前一个实例。示范：每次移动后和按按钮后的画面。
"""

from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec

SPEC = TaskSpec(
    kind=FIXED,
    steps=("Move one block into the empty tray.", "Move a block into the tray that was just emptied.",
           "Move the last block into the tray that was just emptied.", "Press the button."),
    merge_into_previous=("Use right arm to pick up the right block and move it to the right tray.",),
)
