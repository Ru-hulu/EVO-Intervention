SOC35 RoboBrain2.5-4B and Robo-Dopamine-GRM-3B training caches

Preparation only. No training, evaluation, model forward, or GPU allocation.
Reuse the exact preceding FoREST/Robometer/ProcVLM 35-per-suite subsets.
Validate every current manifest task is represented at least twice, all selected
episodes belong to train and not val, and source videos, RGBs, labels, and frame
indices match the verified 5 FPS caches (including the original endpoint).

Separate model-native CPU processors/pixel limits. Eight images: initial/final
main references, initial three views BEFORE, current three views AFTER. Future
endpoint use is explicit. Missing wrist views retain the established black RGB.
Keep continuous PT unchanged and also provide native integer-percent <score>
assistant targets (same semantics as the approved Recovery fine-tuning).
No invented CoT; prompt/image tokens receive no loss.

Run prepare, then build --model MODEL --suite State --smoke for each model,
then controller. Controller runs six CPU-only jobs and an independent all-row
audit. Per-episode receipts permit resume after verifying no old worker remains.
Never overwrite committed episode caches. Data/weights/results stay remote.

Setup: /cache/cvpzhr/setup/soc35-grm-caches-20261115
Data: /cache/cvpzhr/datasets/soc35-grm-5fps-20261115
Each MODEL/SUITE/train.jsonl has a corresponding data-binding.json, documenting
the existing Dataset/CachedCollator and official model revision/pixel settings.
Completion requires final-audit.json status passed, not just exited workers.
