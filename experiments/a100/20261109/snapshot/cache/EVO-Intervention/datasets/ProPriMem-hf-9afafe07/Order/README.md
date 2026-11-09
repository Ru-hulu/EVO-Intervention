# Order

Progress depends primarily on the current position in an action sequence.

14 tasks · 551 trajectories.

| Task | Original task | Source dataset | Domain | Trajectories | Views |
|---|---|---|---|---:|---|
| [Open and close drawers in sequence](01-open_close_drawers_in_sequence/) | [Pull open and close drawers 2](https://huggingface.co/datasets/wzlllong/value-model-benchmark-training-data/blob/458b08ae82e34da4dfc76070ff1056a3d5eab057/data/real-world/full-shot/meta/tasks.parquet) | [value-model-benchmark-training-data](https://huggingface.co/datasets/wzlllong/value-model-benchmark-training-data) | Real robot | 50 | main_view, left_view, right_view |
| [Enter a number sequence](02-enter_number_sequence/) | [Press the numeric keypad](https://huggingface.co/datasets/wzlllong/value-model-benchmark-training-data/blob/458b08ae82e34da4dfc76070ff1056a3d5eab057/data/real-world/full-shot/meta/tasks.parquet) | [value-model-benchmark-training-data](https://huggingface.co/datasets/wzlllong/value-model-benchmark-training-data) | Real robot | 50 | main_view, left_view, right_view |
| [Swap blocks with confirmation](03-swap_blocks_with_confirmation/) | [Swap the two blocks using the empty mat, pressing the button after each move.](https://huggingface.co/datasets/wzlllong/value-model-benchmark-training-data/blob/458b08ae82e34da4dfc76070ff1056a3d5eab057/data/simulation/full-shot/meta/tasks.parquet) | [value-model-benchmark-training-data](https://huggingface.co/datasets/wzlllong/value-model-benchmark-training-data) | Simulation | 50 | main_view, left_view, right_view |
| [Play the xylophone](04-play_xylophone/) | [play_Xylophone](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/269457a3b42d8d0ae69898a601f2c1487c2f826e/data/RoboDojo/play_Xylophone) | [RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) | Simulation | 50 | main_view, left_view, right_view |
| [Brush a water bottle](05-brush_water_bottle/) | [Brush water bottle](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/blob/2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278/task_info/task_392.json) | [AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | Real robot | 50 | main_view, left_view, right_view |
| [Disinfect shelves](06-disinfect_shelves/) | [Disinfect the shelves with a sanitizing gun.](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/blob/2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278/task_info/task_578.json) | [AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | Real robot | 50 | main_view, left_view, right_view |
| [Iron clothes](07-iron_clothes/) | [Iron clothes](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/blob/2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278/task_info/task_440.json)<br>[Ironing Clothes](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/blob/2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278/task_info/task_475.json) | [AgiBotWorld-Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | Real robot | 14 | main_view, left_view, right_view |
| [Enter the PIN sequence 1-3-3-7](08-enter_pin_sequence/) | 按密码 | EvoMind-PiperX | Real robot | 30 | main_view, left_view, right_view |
| [Press buttons in a hidden order](09-press_buttons_in_hidden_order/) | 按隐藏顺序操作按钮 | EvoMind-PiperX | Real robot | 36 | main_view, left_view, right_view |
| [Tap cups in sequence](10-tap_cups_in_sequence/) | 按顺序敲击杯子 | EvoMind-PiperX | Real robot | 35 | main_view, left_view, right_view |
| [Move blocks in sequence and return them](11-move_blocks_in_sequence_and_return/) | 按顺序移动积木并归位 | EvoMind-PiperX | Real robot | 34 | main_view, left_view, right_view |
| [Open and close drawers from top to bottom](12-open_close_drawers_top_to_bottom/) | 按顺序开关抽屉 | EvoMind-PiperX | Real robot | 35 | main_view, left_view, right_view |
| [Flip cups from left to right](13-flip_cups_left_to_right/) | 从左到右翻转杯子 | EvoMind-PiperX | Real robot | 32 | main_view, left_view, right_view |
| [Inspect cup bottoms from left to right](14-inspect_cup_bottoms_left_to_right/) | 从左到右检查3个杯子底部 | EvoMind-PiperX | Real robot | 35 | main_view, left_view, right_view |

O08–O14: recorded on 2026-10-31 with bimanual PiperX, 1280×720 at 30 FPS. All 237 trajectories are unannotated (`has_annotation=false`); task instructions are preserved in `source_instruction`. The main view is the environment camera, and left/right views are wrist cameras. Videos were cut at exact episode frame boundaries and losslessly encoded to H.264, with decoded pixels checked against source cuts.
