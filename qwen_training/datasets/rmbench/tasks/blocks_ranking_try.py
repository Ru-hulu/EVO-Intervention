# ruff: noqa: RUF002
"""试错：先按按钮检查排列，不对就交换两块再按，直到排列正确。

按按钮比交换多一次；50 条中最多交换 5 次、按 6 次。
示范：一次按按钮、一次交换，以及成功时的画面。
"""

from qwen_training.datasets.rmbench.spec import SEARCH, TaskSpec

SPEC = TaskSpec(
    kind=SEARCH,
    steps=("Press the button to check the arrangement.", "Swap two blocks."),
    step_prefixes=(("Press", 0), ("Swap", 1)),
)
