# RoboBrain2.5-4B: fixed-start forward GRM adaptation

This is an offline, same-episode-goal-conditioned experiment. The user explicitly selected the episode's true first and last frames as reference anchors. The eight image slots are reference start main, reference end main, first-frame main/left/right, current-frame main/left/right. Intermediate history and Forest memory features are not inputs.

Native GRM text generation is retained. The assistant target is `<score>{100 * existing_PT:+.1f}%</score>`. Existing floating-point PT remains the evaluation target; answer rounding introduces at most 0.0005 PT quantization error. Static and jitter observations retain anchor PT. No labels are inferred from the augmented frame number or replay-frame PT.

The data exporter reuses the existing 1,656 training episodes, 417 clean validation episodes and the exact 417 augmented validation timelines. The partial-view O03 episode remains excluded. Native frame 0/PT 0 and the true native final frame/PT 1 are checked for all 2,073 episodes; these checks do not independently establish semantic task success. Missing whole auxiliary views use the existing black 64x64 convention.

Raw image files and compatible Qwen3-VL preprocessed pixel tensors are reused. Text/image token sequences and assistant-only loss masks are rebuilt for this interface. Pixel equivalence is checked against the official RoboBrain processor on every task and augmentation kind. No source files, base weights, existing PT, or previous experiment outputs are modified.

Official model: BAAI/RoboBrain2.5-4B, revision ec4b3bf5e969b48ad17df432b479c10402a0079b. The prompt is adapted from FlagOpen/Robo-Dopamine's `dataset/utils/1_generate_data.py`; `upstream.json` records provenance and changes. The native Qwen3VLForConditionalGeneration class is selected explicitly instead of guessing the architecture from the directory name.

`config.draft.json` contains candidate hyperparameters for a future training launch. They are not a user-confirmed training schedule. The adaptation task prepares and tests the interface; it does not start the full training run. `smoke-adapter/` is only a disposable implementation check and must not initialize formal training. A formal run initializes fresh language LoRA from official weights; the vision encoder, base language weights and LM output head remain frozen. No new PT regression head is added.

Training uses native causal-language-model cross-entropy on assistant answer tokens. Clean validation uses target-free generation, frame-weighted MAE and episode-equal Pearson. A complete adapter/optimizer/scheduler/RNG checkpoint is saved before each validation; best is chosen by clean-validation MAE. Invalid model responses fail scheduled validation explicitly rather than being silently scored as zero. Standalone evaluation records invalid responses and withholds full-coverage metrics when any response is invalid.

Entrypoints (run from this directory using `/cache/cvpzhr/envs/robometer/bin/python`):

- `prepare.py`: create compact data manifests once.
- `checks.py`: verify coverage, labels, processor parity and target-independent inference.
- `smoke.py`: four isolated single-GPU optimization steps, generation and adapter save/load.
- `train.py --config /cache/cvpzhr/setup/robobrain25-forward-20261105/config.draft.json`: explicit training entry (candidate settings).
- `evaluate.py --config ... --adapter ... --split augmented_validation --output-dir ...`: generation-based evaluation; omit adapter to evaluate official base weights.

These results must be labeled as using the same-episode final-frame reference. Existing Forest and Robometer results do not have that additional goal image.
