---
pretty_name: ProPriMem-State
tags:
  - robotics
  - memory
  - state
---
# ProPriMem-State

19 State tasks · 941 trajectories · 2,505 videos · 618 stage-annotation files.

| ID | Task | Trajectories | Source / domain |
|---|---|---:|---|
| S01 | [Cover and uncover blocks](State/01-cover_uncover_blocks/) | 50 | MINT-SJTU/ProPriMem · simulation |
| S02 | [Put a block back](State/02-put_back_block/) | 50 | MINT-SJTU/ProPriMem · simulation |
| S03 | [Move blocks to a green mat and back](State/03-place_block_mat/) | 50 | MINT-SJTU/ProPriMem · simulation |
| S04 | [Try battery orientations](State/04-battery_orientation_trials/) | 50 | MINT-SJTU/ProPriMem · simulation |
| S05 | [Try block arrangements](State/05-block_arrangement_trials/) | 50 | MINT-SJTU/ProPriMem · simulation |
| S06 | [Rearrange blocks](State/06-rearrange_blocks/) | 50 | MINT-SJTU/ProPriMem · simulation |
| S07 | [Pick and place a bowl once](State/07-pick_place_bowl_once/) | 50 | MINT-SJTU/ProPriMem · simulation |
| S08 | [Pick and place a bottle once](State/08-pick_place_bottle_once/) | 98 | MINT-SJTU/ProPriMem · simulation |
| S09 | [Swap two bowls](State/09-swap_two_bowls/) | 88 | MINT-SJTU/ProPriMem · simulation |
| S10 | [Rotate three bowls](State/10-rotate_three_bowls/) | 82 | MINT-SJTU/ProPriMem · simulation |
| S11 | [Put three coins into a piggy bank](State/11-put_three_coins_into_piggy_bank/) | 32 | EvoMind-PiperX · real robot |
| S12 | [Inspect four objects and return them](State/12-inspect_four_objects_and_return/) | 35 | EvoMind-PiperX · real robot |
| S13 | [Inspect three drawers](State/13-inspect_three_drawers/) | 35 | EvoMind-PiperX · real robot |
| S14 | [Weigh four objects and return them](State/14-weigh_four_objects_and_return/) | 35 | EvoMind-PiperX · real robot |
| S15 | [Inspect three test tubes](State/15-inspect_three_test_tubes/) | 49 | EvoMind-PiperX · real robot |
| S16 | [Press five buttons in any order](State/16-press_five_buttons_in_any_order/) | 35 | EvoMind-PiperX · real robot |
| S17 | [Pour water into three cups](State/17-pour_water_into_three_cups/) | 33 | EvoMind-PiperX · real robot |
| S18 | [Inspect five upside-down cups](State/18-inspect_five_upside_down_cups/) | 34 | EvoMind-PiperX · real robot |
| S19 | [Shake three sealed bottles](State/19-shake_three_sealed_bottles/) | 35 | EvoMind-PiperX · real robot |

## Format and splits

Each task under `State/` retains its original `metadata.json`, `metadata.jsonl`, videos and,
when present, stage annotations. Paths in each trajectory index are task-relative. Resolution,
frame rate and available camera views are preserved per trajectory; they are not uniform
across all source datasets. All views of a trajectory share its frozen split.

Dataset v1.2.0 · Split v1.2.0 · **750 train / 191 val**. Explicit sample lists in
[manifest.json](manifest.json) are authoritative. All existing assignments are preserved.

## Sources and annotation status

S01–S10 are copied from [MINT-SJTU/ProPriMem](https://huggingface.co/datasets/MINT-SJTU/ProPriMem/tree/e04d7f2261cbff363b8ec3809a60cd5e6f2c7df7/State):
618 simulation trajectories with their original stage annotations. S01–S06 originate from
[TianxingChen/RMBench](https://huggingface.co/datasets/TianxingChen/RMBench); S07–S10 originate
from [htrbao/LIBERO-MEM_lerobot](https://huggingface.co/datasets/htrbao/LIBERO-MEM_lerobot).
Original source links, instructions, hashes and domain labels remain in each trajectory index.
The MINT repository is unchanged by this copy.

S11–S19 contain 323 real-robot PiperX trajectories collected on 2026-10-31, 2026-11-02 and
2026-11-03, with 969 videos at 1280×720 and 30 FPS. These recordings are **unannotated**:
`has_annotation=false`; stage boundaries and task-success labels are not inferred. The environment
camera is `main_view`, with wrist cameras `left_view` and `right_view`. Task names have no State
prefix. Raw robot state/action Parquet files are not included.

## Selection and validation

User-reviewed exclusions remain unchanged: piggy-bank batch `20261031_153227` #3/#15/#16 and
batch `20261031_161309` #18/#19/#20; tea batch `20261103_134047` #0/#1; upside-down-cup batch
`20261103_142808` #0. These refer to original source episode indices. The bottle task describes
three bottles. The test-tube task combines valid batches of 11 + 3 + 35 trajectories.

The 969 PiperX videos are frame-exact H.264 stream-copy cuts, checked by full decoding against
source frame counts, timestamps and decoded YUV pixels. Imported State files are copied without
re-encoding or changing annotations; hashes and frozen split assignments are verified.
