# Flip-cup recovery: Forest and Robometer, original initialization

Reuses the completed pencil experiment settings; only the task dataset and derived steps change.

- Forest: fresh Qwen3.5-4B, new LoRA/history projection/progress head; GPUs 0 and 1; microbatch 3, accumulation 3.
- Robometer: official Robometer-4B revision beef63bc914c5c189329d49c6d712d96d632aa34, new LoRA and official progress head; GPUs 2 and 3; batch 9 per GPU.
- Both: effective batch 18, 3 epochs, learning rate 1e-4, 3% warmup and cosine, seed 42.
- 28 training trajectories / 7,769 observations; 7 validation trajectories / 1,744 observations, 5 FPS. Original PT including real rollbacks is preserved; no artificial augmentation.
- 432 optimizer steps per epoch / 1,296 total. Save each epoch before any validation, evaluate only epoch 3. Frame-equal MAE and episode-equal Pearson, with undefined counts.
- Four-step smoke weights are discarded before fresh formal initialization. Existing experiment outputs are preserved.

The main branch contains the Robometer implementation; externmem-histoken-streamforest contains Forest. Shared orchestration expects both subdirectories in the deployed setup. Source configurations and runtime scripts are archived; datasets, cached tensors, model weights, predictions, and training outputs are excluded.

Remote setup: /cache/cvpzhr/setup/forest-robometer-flipcup-official-e3-20261109
The pipeline has already been launched in the background. Do not relaunch an active controller.
