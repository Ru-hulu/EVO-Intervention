"""Per-task annotation semantics and rewind approval.

No task has been approved for rewind. Source action segments are counted as
annotated segments, never silently presented as physical action repetitions.
"""

# task_id: (schema, original RMBench task or None, rewind_safe)
POLICIES = {
    "C01": ("number_buttons", None, False),
    "C02": ("blue_button", None, False),
    "C03": ("drum", None, False),
    "C04": ("soap", None, False),
    "C05": ("season", None, False),
    "C06": ("rmbench", "press_button", False),
    "C07": ("libero", None, False),
    "C08": ("libero", None, False),
    "C09": ("libero", None, False),
    "C10": ("libero", None, False),
    "C11": ("cube_place", None, False),
    "C12": ("cube_swing", None, False),
    "C13": ("source_segments", None, False),
    "C14": ("source_segments", None, False),
    "O01": ("drawers", None, False),
    "O02": ("keypad", None, False),
    "O03": ("swap_confirm", None, False),
    "O04": ("xylophone", None, False),
    "O05": ("source_segments", None, False),
    "O06": ("source_segments", None, False),
    "O07": ("source_segments", None, False),
    "S01": ("rmbench", "cover_blocks", False),
    "S02": ("rmbench", "put_back_block", False),
    "S03": ("rmbench", "place_block_mat", False),
    "S04": ("completed_counts", "battery_try", False),
    "S05": ("completed_counts", "blocks_ranking_try", False),
    "S06": ("rmbench", "rearrange_blocks", False),
    "S07": ("libero", None, False),
    "S08": ("libero", None, False),
    "S09": ("libero", None, False),
    "S10": ("libero", None, False),
}
