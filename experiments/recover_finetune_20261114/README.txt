# Recover official-model fine-tuning

This workspace owns GPU 0/1 and the sequential order **RoboMeter → ProcVLM →
RoboBrain2.5 → Robo-Dopamine**. FoREST is owned by the other workspace on GPU 2/3;
VLAC is not requested for training. No evaluation or recurring automation is started.

The verified Recover cache selects two original training episodes per task, seed 42:
10 episodes / 2,550 observations at 5 FPS. Existing source videos, original PT,
validation split, view/history inputs and image preprocessing remain unchanged.
All runs start independently from their published weights with fresh rank-16,
alpha-32, dropout-0.05 language LoRA. Each runs one epoch.

| Method | Trainable objective | Initial microbatch/rank | Accumulation | LR |
|---|---|---:|---:|---:|
| RoboMeter | Native continuous-PT C51 progress-head loss; language LoRA + published progress head | 8 | 1 | 1e-4 |
| ProcVLM | Assistant `<progress>` text and EOS only; no CoT, remaining-action or auxiliary-head loss | 2 | 4 | 1e-5 |
| RoboBrain2.5 | Native `<score>` text and EOS only | 8 | 1 | 1e-4 |
| Robo-Dopamine | Native `<score>` text and EOS only | 8 | 1 | 1e-4 |

RoboMeter uses the existing uniform-history objective. Its initial batch 9 was
reduced to 8 before any formal optimizer update after a mixed-length batch hit
the 78% allocator cap (16.5 GiB of device memory remained free). ProcVLM keeps
its prior LR and linear schedule/5% warmup; the other official models use
cosine/3% warmup. ProcVLM retains its native high-resolution eight-image input
(up to 7,163 tokens including the answer), without truncation. The two GRMs
retain true future endpoint references. Their BEFORE reference is the initial
PT=0 state, so the native relative completion score equals 100×current PT.
Their integer-percent instruction is honored by round-half-up text targets;
raw continuous PT is retained separately, with at most 0.005 PT quantization.

`prepare` freezes configs and source/weight fingerprints; `queue` uses a process
lock and checks GPU 0/1 have no compute contexts before each job. It runs a
three-update pilot mixing longest and shortest real inputs before formal training
(to exercise padded attention kernels). Pilot
weights are discarded. Formal training reloads official weights and fresh LoRA.
Unknown failures halt the queue with all outputs retained; no blind retry.

Each model saves language adapters, any native progress head, exact trainable
tensors, optimizer, scheduler, both ranks' RNG and Trainer cursor. Final checks
require the expected epoch/step count, finite changed parameters, identical rank
weights and matching input/selection hashes. Intermediate checkpoints support an
explicit audited restart; a finished model releases both cards before the next.

The allocator is capped at 78%, with 16 GiB minimum GPU headroom. Cgroup safety
checks count RSS/shmem, dirty/writeback/unevictable/mapped pages and OOM events.
Clean active file cache is reclaimable and is not treated as anonymous memory.
Resource stops affect only this queue and preserve a checkpoint where possible.
