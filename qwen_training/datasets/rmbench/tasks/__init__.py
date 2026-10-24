# ruff: noqa: RUF002
"""任务名 → TaskSpec，一个任务一个文件。"""

from qwen_training.datasets.rmbench.tasks import (
    battery_try,
    blocks_ranking_try,
    classify_blocks,
    cover_blocks,
    observe_and_pickup,
    place_block_mat,
    press_button,
    put_back_block,
    rearrange_blocks,
    storage_blocks,
    swap_blocks,
    swap_T,
)

TASKS = {module.__name__.rsplit(".", 1)[1]: module.SPEC for module in (
    battery_try, blocks_ranking_try, classify_blocks, cover_blocks, observe_and_pickup, place_block_mat,
    press_button, put_back_block, rearrange_blocks, storage_blocks, swap_blocks, swap_T)}
