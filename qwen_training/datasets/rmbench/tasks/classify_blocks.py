# ruff: noqa: RUF002
"""固定流程：6 个方块逐个放进对应颜色的篮子。

每 4 段（移动并抓取、提起、移到篮子上方、松手）放进一个方块。两种颜色各几块每条不同，
所以按"第 k 个方块"对应，不按颜色。示范：每个方块放入后的画面。
"""

from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec

SPEC = TaskSpec(
    kind=FIXED,
    steps=tuple(f"Put the {n} block into the basket of its color."
                for n in ("first", "second", "third", "fourth", "fifth", "sixth")),
    segments_per_instance=4,
)
