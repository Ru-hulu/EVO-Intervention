# S01 declining-progress fine-tuning experiments

Huawei registry: `/cache/cvpzhr/experiments/forest-s01-decline/index.json`.

| Run | Sampling | Results |
| --- | --- | --- |
| `01-uniform-20261106` | Original uniform frame sampling | Completed 13,224 updates; full-val MAE 1.3242 pp, RMSE 7.6607 pp |
| `02-critical-sampling-20261106` | Critical abnormal intervals oversampled | Started 2026-11-06 18:09 CST on GPUs 0–3; results pending |

Each registry entry points to its manifest under the same experiment directory. The baseline archive contains weights, per-epoch checkpoints, training/validation logs, config, split and references to the preserved five- and ten-trajectory evaluations. The new run archives its config, split, sampling audit and code hashes before training; its controller archives final checkpoints/logs and generates `loss.png` after training finishes. Original output and evaluation directories remain intact.

## New sampling policy

Keep normal frames at 26.24% of sampling probability and abnormal frames at 73.76%. Within abnormal frames, allocate half the probability uniformly and half equally among wrong-approach, failure-transition, moving-after-failure and recovery groups. Recovery counts as abnormal. Critical intervals increase from 13.34% of original entries to approximately 43.55% of expected draws. Paused copies remain in the uniform component and receive no critical boost.

Groups use training annotations and existing visual-motion deltas only. Validation episodes and labels are unchanged. Sampling is with replacement; each epoch draws 35,258 entries with a deterministic epoch seed, then shards the same global draw across four ranks. Cached causal histories are retained. Audits check shard reconstruction, zero-weight tail padding, determinism, data signatures and source-label agreement.

## Training and comparison

Both runs initialize model weights from `/cache/cvpzhr/outputs/forest-original-cap50-static-jitter-e3-20261104/last.pt`, with optimizer and training counters reset. The new run does not resume the completed S01 model. Both use the v3 declining-progress labels, the same seed-grouped train/val split, global batch 8, learning rate 1e-5 and 13,224 updates over three epochs. The new run uses four GPUs throughout; the baseline migrated from two to four GPUs, so this is not a bitwise-controlled comparison.

New setup: `/cache/cvpzhr/setup/forest-s01-critical-sampling-20261106`.
New outputs: `/cache/cvpzhr/outputs/forest-s01-critical-sampling-20261106`.
Gradient checkpointing is disabled. Full validation runs every 2,204 updates. New validation also logs cross-entropy and errors by event type; the baseline validation did not log cross-entropy. Training CE across differently sampled distributions is not directly comparable; compare on the unchanged validation set.

Check `status.json` and `train.log` in the setup directory for live status. `last.pt` is saved every 200 updates; `best.pt` is selected by full-validation MAE, and epoch checkpoints are retained. The launch controller owns automatic final archiving.
