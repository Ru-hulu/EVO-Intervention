# IMPRINT × FOREST: online DSRL on RMBench Cover Blocks

**Status: active online DSRL route.** See the [current plan](../../../docs/imprint_dsrl_plan.md).
Current groups are A = original IMPRINT, B = terminal-success DSRL,
C = terminal-progress DSRL. The historical A/B labels in the smoke records below
mean terminal/terminal_progress respectively; those records remain unchanged.
B does not require FOREST. C requires reward validation and an explicit lambda
interface (current smoke code uses lambda=1; the new plan proposes 0.25).

This integration imports the **actual RoboMeter SAC, ReplayBuffer, MLPActor and
MLPCritic** implementation, pinned to `robometer/robometer-policy-learning`
commit `4fe3476230befaea6ec6fd84c5a8a24eae049669`. It uses a small serial
coordinator for RMBench instead of RoboMeter's Pi0/LIBERO-specific training CLI.
It does not reimplement SAC, retrain IMPRINT, or replace its memory mechanism.

## Controlled reward comparison

| Run | Intermediate reward | Terminal reward |
| --- | --- | --- |
| A: `terminal` | 0 | Native RMBench success = 1, failure/deadline = 0 |
| B: `terminal_progress` | FOREST `p_next - p_current` | Same native success bonus as A, plus final progress difference |

A **does not load or call FOREST**. Both policies receive the same frozen IMPRINT
features: pooled recurrent memory read, masked recent visual embedding, language
embedding, current joint state and remaining native horizon. No simulator object
poses, correct-lid identifiers, `fail_flag`, success label, or FOREST prediction
enters the actor observation. `fail_flag` is recorded for audit only.

Both runs start from the same actor initialization and use the same training and
evaluation seed ranges. They have separate actors, critics, optimizers, replay
buffers, logs, and checkpoints. Scene seeds accepted by the native expert
preflight are recorded; check actual seeds/instructions when comparing results.

The noise action is 32-dimensional, bounded to [-1, 1], repeated across the
50-step Pi0.5 noise horizon exactly as in RoboMeter's `pi0_infer_with_noise`.
IMPRINT produces the ordinary 50 × 16 action chunk. Its original action mapping,
memory writes, instruction generation, success checks, and step limit remain in
use. Rendering uses the previously selected 8 samples/pixel and three cameras.

For this finite-horizon smoke experiment, gamma = 1. The native deadline is a
terminal task outcome, not a time-limit truncation to bootstrap through. Episode
return is exactly `success` for A and `success + p_final - p_initial` for B.
Negative progress differences are retained. There is no reward from a reset,
no clipping of negative changes, and no ground-truth failure override of FOREST.
This defines B's objective explicitly; it is not a claim of policy-invariant
potential shaping. A plateau contributes zero dense reward, not a waiting penalty.

FOREST is frozen at an immutable snapshot. The model gets raw camera arrays in
the same channel convention as `capture.py` training videos, three current views,
and the current task instruction. Head-camera memory updates every four native
observations (the IMPRINT dataset's 20 fps convention, sampled at 5 Hz). Progress
is queried at action-chunk boundaries. Current images are excluded from their
own history. Extra duplicate evaluator renders are not added as new control steps.

## Huawei setup

- Source: `/cache/robot-baselines/setup/imprint-dsrl-20261106`
- Upstream: `/cache/robot-baselines/repos/robometer-policy-learning`
- CPU learner: `/cache/robot-baselines/envs/imprint-dsrl/bin/python`
- Frozen policy: existing `envs/imprint`; simulator: existing `envs/rmbench`
- Frozen reward: existing `/cache/cvpzhr/envs/forest`

The isolated CPU learner reuses the installed PyTorch package read-only through
`shared-torch.pth`, and installs `gymnasium`, `loguru`, `omegaconf`, and `pytest`
locally. This avoids changing the existing GPU environments. The upstream
full-stack package declares Python 3.11; this adapter tests its SAC/core subset
under the host's Python 3.12, while IMPRINT/simulation stay in their original
Python 3.10 environments. It does not claim validation of upstream's full CLI.

Run unit checks (actual SAC gradient update, terminal masking, reward cases,
save/load including optimizer state):

```bash
cd /cache/robot-baselines/setup/imprint-dsrl-20261106
CUDA_VISIBLE_DEVICES= PYTHONPATH=/cache/robot-baselines/repos/robometer-policy-learning \
  /cache/robot-baselines/envs/imprint-dsrl/bin/python -m pytest test_core.py -q
```

Run each smoke experiment in a **new** output directory:

```bash
bash /cache/robot-baselines/setup/imprint-dsrl-20261106/launch.sh \
  --reward-mode terminal --policy-gpu 0 --port 18400 \
  --output /cache/robot-baselines/eval-results/imprint-dsrl-20261106/A-new

bash /cache/robot-baselines/setup/imprint-dsrl-20261106/launch.sh \
  --reward-mode terminal_progress --policy-gpu 2 --forest-gpu 1 --port 18410 \
  --output /cache/robot-baselines/eval-results/imprint-dsrl-20261106/B-new
```

Default: one full training episode starting at seed 310000, followed by one
evaluation episode starting at seed 320000. `--train-episodes`, `--eval-episodes`,
`--train-seed`, and `--eval-seed` configure larger runs. Do not overwrite existing
runs. For paired B runs, pass `--forest-checkpoint` to the same immutable snapshot.

## Acceptance and artifacts

`result.json` is written only after native episodes finish, the actor parameters
change, a checkpoint is saved and reloaded identically, and the saved actor
completes evaluation with training disabled. Success comes from RMBench, never
from a threshold on progress. A single smoke episode is **not** a reliable
success-rate estimate or evidence of improvement over the existing IMPRINT baseline.

- `manifest.json`: settings, upstream commit, reward checksum, seed ranges.
- `code/`: immutable source snapshot actually used by child processes.
- `learner/events.jsonl`: p before/after, separate terminal/dense rewards,
  native outcomes, actor/critic losses, update counts, and timings.
- `learner/last.pt`: actor, critics/target, optimizers, replay, RNG and provenance.
  Restores at episode boundaries; live simulator state is not checkpointed.
- `train/`, `eval/`: canonical trials and three-view videos/state/action traces.
- `error.json`: incomplete integration attempts, excluded from success rates.

The launcher owns and cleans up only its subprocess groups. Existing training,
datasets, checkpoint sources, and visualization tools are untouched.

## Verified smoke result — 2026-11-06

Both `terminal-smoke04` and `terminal_progress-smoke04` under
`/cache/robot-baselines/eval-results/imprint-dsrl-20261106/` completed the full
rollout → SAC update → save → reload → held-out rollout loop. Four unit tests
passed. Canonical trial records, captured evidence, and reward accounting agree.
Both runs released their subprocesses and GPU memory after completion.

| Arm | Train seed 310000 | SAC updates | Eval seed 320000 |
| --- | --- | --- | --- |
| A: terminal | Failure, 1500 steps | 27 | Success, 925 steps |
| B: terminal + progress | Success, 944 steps | 16 | Success, 938 steps |

This is one training and one evaluation episode per arm, not evidence that either
method improves success rate. The accepted scene seeds and instructions match.
The B reward snapshot is FOREST step 8816, SHA256
`dfe5d9e6fba7010e27181e4aca6d4de5c077034aa4ccb2468f3bcb89cf386772`.

**Observed reward limitation:** FOREST predicted only 0.1009 at B's successful
training endpoint and 0.00761 at its successful evaluation endpoint. On the last
evaluation chunk, p fell from about 0.929 to 0.00761 despite native success. The
integration preserves this prediction rather than silently correcting it with
ground truth. This needs reward-model investigation before claiming reliable
dense-reward RL. The separate terminal-success reward remains authoritative.

`acceptance.json` in the run root records cross-checks. Earlier `smoke01`–`03`
folders are incomplete integration attempts and are excluded from task metrics.
The source under each run's `code/` is the executed snapshot. The current launcher
also fixes unused FOREST-path metadata in arm A and handles SIGTERM cleanup;
these do not change the verified reward or policy computation.
