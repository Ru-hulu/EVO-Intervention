Original Qwen3.5-4B for visual progress labels

The official `Qwen/Qwen3.5-4B` snapshot is pinned to
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` in
`qwen35-download-catalog.json`. The catalog includes all 14 repository files,
9,342,907,469 bytes, with the official Git blob SHA1 or LFS SHA256.

Run on A100 from the repository root:

```bash
python -m experiments.proprimem_visual_pt_20261113.download_qwen35 \
  --state-dir /cache/cvpzhr/setup/qwen35-original-download-20261113 \
  --workers 4
```

All model files download directly on A100 to
`/cache/cvpzhr/models/Qwen3.5-4B/`. The downloader resumes partial files,
holds an exclusive job lock, verifies each official hash before publishing a
file, and checks that the weight index references verified shards. Large files
prefer the author's faster ModelScope CDN; every file must match the pinned
official Hugging Face hash. No Hugging Face token is sent to either mirror.

Completion and per-file hashes are recorded in `status.json` and
`verification.json` in the state directory. This command does not remove time
labels, generate new labels, or start training or evaluation. Model weights,
datasets, labels, and runtime status files are not stored in Git.

Visual progress relabeling (explicitly authorized separately)

The relabel.py command removes only normalized_time_v1 progress fields from
State, Order, and Count, preserving action/stage annotations. The expected
removal inventory is State 441, Order 50, Count 32. It processes all 1,904
trajectories / 1,651,148 original main-view frames. It uses the existing Forest
extract_deltas, whole_video_progress, and load_vision_encoder functions without
changing the mathematics: frozen BF16 Qwen3.5 vision/merger, FP32 mean features
and adjacent L2, FP64 cumulative sum divided by its final value. No resampling,
stage weighting, time interpolation, epsilon addition, or rollback is used.
Zero/tiny visual change fails explicitly.

The dedicated /cache/cvpzhr/envs/forest-labels environment pins Torch 2.10.0,
Transformers 5.3.0 and PyAV 19.0.2. Four GPUs run four worker processes each,
batch size 16, min_pixels 4096, max_pixels 262144, SDPA. A transactional SQLite
queue assigns longer videos first and gives each video to exactly one worker.
This overlaps CPU decoding/preprocessing across jobs while keeping each
video's delta computation identical to Forest's original routine.

Run from the A100 code root only after confirming all four GPUs are free:

python -m experiments.proprimem_visual_pt_20261113.relabel run

State: /cache/cvpzhr/setup/proprimem-visual-pt-20261113/status.json
Per-video output: <suite-task-directory>/progress/episode_XXXXXX.progress.csv
Columns: frame_index,delta,pt. An adjacent .progress.json records provenance.

The pipeline records the original manifest splits and annotation hashes,
removes time fields with a hash-only audit (no retained copies of old PT
arrays), and hashes all Recover files before and after. After complete CSV
validation it adds progress_path/progress_sha256/progress_policy to each
trajectory metadata row and refreshes the manifest index hashes. It preserves
all train/val memberships. All 16 initialization audits must agree on visual
weights. Complete coverage, original frame indices, numeric validity, and the
normalization formula are checked for every video before publishing readiness.
The final state is completed_awaiting_evaluation_notification. No command here
launches evaluation. Review status/processes before any recovery; the run
command intentionally refuses an existing queue or occupied GPUs.

If an independent import adds Recover files during the run, finalization stops.
After reviewing the exact additions and confirming every original Recover file
is unchanged, a hash-pinned additions receipt can be supplied to `finalize` with
`--recover-additions-audit`. It contains `before_snapshot_sha256` and an exact
`added_files` mapping in the original snapshot format. Unlisted additions,
modified additions, or any changed/deleted original file still fail. The final
report distinguishes original-file integrity from the changed file inventory;
no Recover file or original snapshot is rewritten. Finalize takes the pipeline
lock and updates status, without rerunning GPU label extraction.
