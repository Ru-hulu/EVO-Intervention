# ruff: noqa: RUF002
"""试错：按 [正正, 反正, 反反, 正反] 的顺序逐个尝试电池方向，直到仪表盘指针转动。

第一次是双臂把两节电池正向放入；之后每次尝试只翻转一节电池，最多再试 3 次。
示范：第一次放入、一次翻转，以及成功时的画面。
"""

from qwen_training.datasets.rmbench.spec import SEARCH, TaskSpec

SPEC = TaskSpec(
    kind=SEARCH,
    steps=("Place both batteries into the slot in the positive direction.",
           "Flip one battery to try the next orientation combination."),
    step_prefixes=(("Use dual arm", 0), ("Pick up", 1)),
    max_counts=(1, 3),
)
