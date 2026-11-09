# RMBench VLA collection

Huawei deployment: `/cache/robot-baselines/setup/proprimem-vla-failures-20261104`.

User's final quantity is **10 complete task episodes per task per model = 210 episodes**, retaining successes and failures. No train/validation split is assigned. The seven tasks and checkpoints match the authenticated HF ProPriMem manifest captured on 2026-11-04. The original evaluation's 10 completed videos are preserved separately.

Three workers run on GPU 0/1/2. Each model remains resident across episodes. Before collection, each worker runs seven same-seed paired 32/8 spp pilots (42 total pilot episodes across workers, excluded from the 210). Gate: same accepted scene seed, native outcome, evidence-derived progress event sequence; action count within 20% or 10 actions; normalized-time 64-point joint-route mean absolute difference <=0.15. This is a pilot consistency check, not a proof of pixel-identical policy behavior. A rejected worker retains 32 spp. No physics, camera resolution, action horizon, model sampling, or official success rules are changed.

Output: `/cache/robot-baselines/eval-results/vla-failures-20261104`.
Viewer export: `/cache/robot-baselines/viewer-datasets/ProPriMem-VLA-Failures-20261104`.
Pilot output: `/cache/robot-baselines/eval-results/vla-render-validation-20261104`.
Logs: `/cache/robot-baselines/logs/vla-failures-{simplemem,full_attention,recurrent}.log`.

Capture saves synchronized head/left/right videos, native joint states, task state evidence and executed actions. It never calls `check_success` as an observer; native evaluator calls retain their original semantics. Redundant third-person and mosaic rendering/recording is disabled. Unused planner child processes are closed between scenes while model parameters remain resident.

Annotations use passive native state and geometry events as conservative progress lower bounds. No elapsed-time ramps or terminal-failure reset. Count overpress and native irreversible cover failure have explicit evidence drops. Battery/permutation labels describe unique tried-state coverage; these and Count are marked `needs_review` and excluded from automatic training eligibility until reviewed. Other rule annotations also require visual spot checks before dataset admission. No training is started by this collector.

`worker.py` pauses after pilots if `collection_config.json` is missing. Config `{"mode":"episodes"}` selects the user's 210-episode quantity. Errors are written to `<model>-error.json` and are never counted as policy failures or valid completed episodes.

Run focused checks on Huawei: `python test_progress.py`. Original upstream evaluation code is left intact; `sim_entry.py` applies scoped capture/video overlay changes only in collection subprocesses.
