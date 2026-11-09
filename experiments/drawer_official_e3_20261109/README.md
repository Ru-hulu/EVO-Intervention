# Drawer inspection: Forest and Robometer, original initialization

The earlier five-task joint Forest run included drawer data. No prior standalone drawer experiment was found for either model. This experiment preserves the completed spray experiment hyperparameters and initialization, changing only the task dataset and derived counts.

- Forest: fresh Qwen3.5-4B, new LoRA/history projection/progress head; GPUs 0 and 1; microbatch 3, accumulation 3.
- Robometer: official Robometer-4B revision beef63bc914c5c189329d49c6d712d96d632aa34, new LoRA and official progress head; GPUs 2 and 3; batch 9 per GPU.
- Both: effective batch 18, 3 epochs, learning rate 1e-4, 3% warmup and cosine, seed 42.
- 21 training trajectories / 3,138 observations; 6 validation trajectories / 719 observations, 5 FPS. Original PT and existing split preserved; no artificial augmentation.
- 175 optimizer steps per epoch / 525 total, 16 warmup steps. Save each epoch before validation; evaluate only epoch 3. Frame-equal MAE and episode-equal Pearson with undefined counts.
- Four-step smoke weights are discarded before fresh formal initialization. Previous experiment outputs are preserved.

Forest caches reuse fully verified original causal base observations with independent metadata and immutable hardlinked payloads. Robometer caches are rebuilt with 32 CPU processes and checked for exact cached/online input equivalence. The model uses 7 past main-view frames plus current 3 views, 10 images total. Cache preparation scripts are included. Data, labels, cached tensors, model weights, predictions, and outputs are excluded from Git.

The main branch contains Robometer; externmem-histoken-streamforest contains Forest. Shared orchestration expects both subdirectories in the deployed setup.
Remote setup: /cache/cvpzhr/setup/forest-robometer-drawer-official-e3-20261109
The pipeline is already launched in the background. Do not relaunch an active controller.
