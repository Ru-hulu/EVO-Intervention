# Flip-cup recovery annotations and progress labels

This directory adds annotations for 35 existing three-view recovery episodes (episode_000000 through episode_000034). It contains 56,802 original 30 FPS frames and 70 progress rollback events.

## Files

- episode_XXXXXX.json: 13-stage action annotations. video_path is relative to the annotation JSON and points to the existing main-view video.
- progress_13stages/episode_XXXXXX.progress.csv: one row per original video frame.
- progress_13stages/episode_XXXXXX.metadata.json: progress rules, milestone weights, rollback events and source hashes.
- progress_13stages/manifest.json and validation.json: original label inventory and validation results.

Paths in progress metadata are relative to that metadata file. All progress CSV files are byte-for-byte copies of the computed labels; PT values and stage boundaries have not been changed for publication. annotation_sha256 hashes the published stage JSON; source_annotation_sha256 records the original JSON before publication path adjustments.

## Annotation quality

Stages were located by AI visual inspection of main-view observations sampled at 5 FPS, then mapped to original 30 FPS frame indices. This mapping does not improve the approximately 0.2-second observation resolution. Some handoff boundaries are accepted approximations, including occluded or out-of-frame handoffs. All 35 records retain human_verified=false. Automatic validation passing does not imply human verification or exact visual boundary timing.

Frame-map, contact-sheet and review-evidence paths in the stage records are provenance references to the original annotation workspace. These auxiliary files are not included here.

## Progress convention

PT ranges from 0 to 1 and can decrease when an error invalidates retained task credit. Each semantic milestone is weighted using the visual-delta sum of its last positive attempt. At the first frame of stage 6, first-cup pickup credit is removed; at the first frame of stage 7, second-cup pickup credit is removed. Stages 6 and 7 earn no positive credit. Stages 8 and 9 jointly restore the first-cup pickup credit in proportion to their delta sums; stage 9 is not a separate positive milestone. Stage 10 restores second-cup pickup credit. Idle frames hold the preceding PT, with initial PT 0 and final PT 1.

Delta is the L2 distance between adjacent original-frame frozen Qwen3-VL-4B post-merger mean features; the first frame has delta 0. The separate delta TXT files are outside this contribution, but every frame's delta is present in the CSV delta column. Metadata therefore sets delta_path to null, provides delta_values pointing to the CSV column, and retains source_delta_filename and delta_sha256 for provenance.

CSV columns: frame_index, time_seconds, stage_id, stage_name, delta, pt, boundary_needs_review. Frame indices are zero-based; stage frame intervals are inclusive. Stage 0 denotes unassigned idle/repositioning frames. A false boundary_needs_review value is not a claim of human verification.

This contribution adds annotations and labels to existing videos. It does not include training caches or model predictions.
