# State / Order / Count: exact FoREST 35-episode selections

Authorized scope: generate Robometer and ProcVLM training caches for each of the
three completed FoREST runs. No new training, resampling, augmentation, PT
normalization, validation split changes, or periodic automation.

Remote setup: `/cache/cvpzhr/setup/robometer-procvlm-soc35-cache-20261114`.
Remote data: `/cache/cvpzhr/datasets/robometer-procvlm-soc35-5fps-20261114`.
Each model has `State/train.jsonl`, `Order/train.jsonl`, `Count/train.jsonl` and
matching `cache-ready.json` receipts. Counts: 6381 / 8783 / 4315, 35 episodes each.
The immutable `selection.json` records the actual completed FoREST configuration,
checkpoint verification receipt, original frame/PT hashes and exact episode IDs.
RGB PNGs are independent hardlinks; no FoREST feature tensors are reused.

## Robometer

Use the frozen `runtime/robometer_training/{model,data,uniform}.py` snapshot.
`ProgressDataset` / `ProgressCollator`: 8 uniform-prefix temporal slots; short
prefixes pad the current frame on the right. Seven sampled main-view slots are
followed by current main/left/right, 10 images and 8 progress tokens. Original
PT is a scalar in [0,1], consumed by the existing C51 soft-label progress loss.
Missing auxiliary images use the existing black RGB64 policy.

## ProcVLM: explicitly authorized progress-only supervision

Use this experiment's `proc_data.ProgressDataset` and `ProgressCollator` for both
training and inference. Do not use the historical `procvlm_training.data` prompt
or remaining-action targets. Current frame plus seven previous 5 FPS main frames;
short prefixes repeat the first frame on the left. No previous CoT input.

Assistant output is exactly `<progress>XX.XX%</progress>` plus the tokenizer EOS.
Only assistant tokens are unmasked for language-model loss. Original full-
precision PT stays in JSON; `progress_gt=100*pt` is supplied as FP32 to the
existing ProcVLM progress head. Formatting to two percentage decimal places
does not replace the numeric label. Every selected observation is eligible.

The experiment-specific collator is compatible with the ProcVLM forward fields
(`labels`, `progress_gt`, normal multimodal inputs). Future training must bind this
collator, not accidentally import the historical remaining-actions collator.
No model forward or optimizer step is performed by cache generation.

## Processing and verification

Both processors use the existing fine-tuning pixel budget: min4096/max65536,
sequence limit4096; original RGB is fed directly to each model's processor.
Store BF16 preprocessed pixels once per observation/view, and gather windows at
load time. Do not run FoREST's history encoder or mix the two model processors.

Each episode verifies cached versus direct preprocessing at its beginning,
padding transition, middle and end, with and without supervision. Inputs, pixel
grids, token IDs, masks and numeric labels must match (pixels compared at BF16).
ProcVLM additionally checks full-text assistant token boundaries/EOS and absence
of supervision fields from inference batches. A final pass compares all 19,479
frame/PT rows for each model against the frozen FoREST source and rechecks source
and progress-file hashes. `cache-ready.json` is written only after these checks.

`controller.py` runs six CPU processes with CUDA hidden and two CPU threads each.
It uses a process lock, persists per-suite status/logs and launches no training.
