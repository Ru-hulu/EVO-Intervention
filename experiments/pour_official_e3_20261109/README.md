# Repeated pouring: Forest and Robometer from original weights

This experiment uses the existing 28-episode repeated-pouring dataset and split: 22 training episodes (6,164 observations), 6 validation episodes (1,437 observations). Images are sampled at 5 FPS including the final frame, with synchronized three-view data. There is no artificial stationary or jitter augmentation. Original PT is preserved: each cup earns progress only during its first pour and associated put-down; repeat visits and idle periods hold PT fixed.

Forest starts from `/cache/cvpzhr/models/Qwen3.5-4B` with fresh LoRA, history projection and progress head. Robometer starts from official `/cache/cvpzhr/models/Robometer-4B` (revision `beef63bc914c5c189329d49c6d712d96d632aa34`), preserving the official progress head and creating fresh language LoRA. Neither model loads a previously fine-tuned checkpoint. Both reset the optimizer, scheduler and counters.

- Two GPUs per model: Forest 0/1, Robometer 2/3.
- Three epochs; effective batch 18; seed 42; learning rate 1e-4, 3% warmup and cosine decay.
- Forest microbatch 3 with accumulation 3; Robometer batch 9 with accumulation 1.
- 343 optimizer steps per epoch, 1,029 total; 31 warmup steps.
- Save epochs 1, 2, 3 at steps 343, 686, 1,029. Validate only the final epoch after saving it.
- Model architecture, native loss, input contracts and other settings are copied unchanged from the drawer comparison. Forest uses native three-view current input plus causal memory. Robometer uses seven past main-view frames plus current three views (ten images).
- Four-step GPU smoke runs are discarded. Formal training starts again from the specified original weights.
- Final metrics: observation-weighted MAE and equally averaged per-episode Pearson, with undefined constant series reported. Plot all six validation trajectories without smoothing.

Remote controller: `/cache/cvpzhr/setup/forest-robometer-pour-official-e3-20261109/pipeline.py`.
Read `status.json` before taking action. Do not launch another controller while this one or its children are alive. It waits for the assigned GPUs to be idle and must not interrupt other jobs.

Forest cache reuses only verified, unaugmented causal base features from the historical five-task preprocessing. Every image/history tensor is audited; cached PT and delta match current source labels. This feature reuse does not initialize model training from a previous experiment. Robometer caches processor inputs from the same original images, not Forest model features.

This archive contains code, configuration and reproducibility metadata only. Videos, PT data, caches, model weights, checkpoints, predictions and plots remain outside Git. Paths are specific to the A100 environment. `configuration-comparison.json` proves only data/path/count/derived-step changes; `preflight.json` records data alignment and official weight checks.
