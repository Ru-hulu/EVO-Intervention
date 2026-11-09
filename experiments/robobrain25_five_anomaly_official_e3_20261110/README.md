# RoboBrain2.5: five independent clean anomaly-task experiments

User-authorized configuration: five tasks, each independently initialized from official BAAI/RoboBrain2.5-4B revision ec4b3bf5e969b48ad17df432b479c10402a0079b. Fresh language LoRA rank16/alpha32/dropout0.05; vision, base language model and LM head frozen. Native GRM answer-token cross-entropy; no new regression head. Three epochs, effective batch18, lr1e-4, warmup3% and cosine, seed42, no artificial static/jitter augmentation. Save all three complete epoch checkpoints before final epoch3-only validation and plots.

The confirmed GRM interface uses eight images: reference start/end main, first-frame three views (BEFORE), current-frame three views (AFTER). Labels use the original PT formatted as a signed percent to one decimal. Evaluation uses unrounded PT and target-free greedy generation. Same-episode true final-frame conditioning is an additional input compared with Forest and Robometer, and must be disclosed in comparisons.

| Task | Training episodes / observations | Validation episodes / observations | Steps per epoch / total |
|---|---:|---:|---:|
| Pencil recovery | 28 / 10,789 | 7 / 2,177 | 600 / 1,800 |
| Flipcup recovery | 28 / 7,769 | 7 / 1,744 | 432 / 1,296 |
| Spray inspection | 25 / 7,416 | 7 / 1,919 | 412 / 1,236 |
| Repeated pouring | 22 / 6,164 | 6 / 1,437 | 343 / 1,029 |
| Drawer inspection | 21 / 3,138 | 6 / 719 | 175 / 525 |

`prepare_cache.py` creates independent image/pixel hardlinks and GRM manifests from the same clean observations used in the current Forest/Robometer comparison. All 157 episodes pass cached-vs-official-processor parity in supervised and inference modes; changing PT/answer cannot change inference inputs. Original labels, split and frames are checked against source and Robometer observations. No Forest memory features are inputs to RoboBrain.

`preflight.py` checks official weight hashes, data coverage and unchanged native GRM implementation. `pipeline.py` runs two lanes on GPU0/1 and GPU2/3 and claims tasks in the order pencil, flipcup, spray, pour, drawer. Each task waits for its assigned pair to be idle; no foreign processes are stopped. Each runs a discarded four-step GPU smoke. Batch9 accumulation1 exceeded memory for the first two tasks; batch3 accumulation3 passed at about46GiB per GPU. Effective batch stays18; candidates and selected execution configs are recorded. Formal model initialization starts afresh after smoke.

`train_run.py` verifies both-rank fresh LoRA initialization, logs training, saves all three complete adapter/optimizer/scheduler/RNG checkpoints and supports exact-configuration resume. `evaluate_final.py` verifies archives and generates final-checkpoint predictions on two independent GPU shards. Existing completed predictions can resume only for the same adapter/config. Malformed native answers are retained and cause full-coverage metrics to be withheld, never replaced with zero. MAE is observation-equal; Pearson is episode-equal with constant-series undefined counts. `plot_final.py` checks plotted ground truth against the original dense PT and produces all validation curves, CSV and ZIP without smoothing.

Remote root: `/cache/cvpzhr/setup/robobrain25-five-anomaly-official-e3-20261110`.
Outputs: `/cache/cvpzhr/outputs/robobrain25-{task}-fromofficial-clean-2gpu-e3-20261110`.
Caches: `/cache/cvpzhr/datasets/robobrain25-{task}-grm8-clean-20261110`.
Environment: `/cache/cvpzhr/envs/robometer/bin/python`.

Read `status.json` and live processes before any action. Never launch a duplicate controller. `pipeline.py --recover-task TASK --gpus G0 G1` is only for a task whose previous controller and all children have stopped, using idle GPUs and a valid same-run complete checkpoint (same config/two GPUs/optimizer/scheduler/RNG). A completed training run can resume final evaluation/plotting without retraining. Raw predictions, caches and checkpoints must be preserved. After an authorized script repair, update preflight code hashes; never silently alter labels, split, initialization, interface, effective batch or learning-rate schedule.

Code/configuration only are archived in Git. Videos, dense PT, input caches, model weights, checkpoints, predictions and plots are not.
