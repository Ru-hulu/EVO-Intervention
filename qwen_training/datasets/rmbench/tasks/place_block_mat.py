# ruff: noqa: RUF002
"""固定流程：从左到右，把每个块移到绿垫上再移回原来的垫子。示范：每个块移回后的画面。"""

from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec

SPEC = TaskSpec(
    kind=FIXED,
    steps=tuple(f"Move the {side} block onto the green mat and back to its original mat."
                for side in ("left", "middle", "right")),
)
