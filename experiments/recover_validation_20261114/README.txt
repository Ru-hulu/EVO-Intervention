Recovery five-model held-out evaluation, 2026-11-14

The user removed VLAC entirely. Models: RoboMeter-4B, ProcVLM-2B,
RoboBrain2.5-4B, Robo-Dopamine-GRM-3B and FoREST. Each uses its final,
verified Recovery ten-training-episode checkpoint. No zero-shot VLAC or R2.

Seed42, sorted task/sample identifiers, two validation trajectories from each
of the original five tasks. Ten held-out episodes, 2132 five-FPS observations;
the last original frame is included. The common selection is disjoint from
all ten training episodes. Original datasets, annotations and splits untouched.

Two missing pouring PT files are separate overlays: existing frozen
Qwen3-VL-4B original-frame deltas, first effective pours and associated put-downs
contribute, repeats and waiting stay flat, final cumulative normalization.

Native prompts, ordered views, history, resolution and greedy output limits
are preserved. Generative adapters are loaded into the same Transformers stack
used for their fine-tuning; no merge, re-quantization, truncation or smoothing.
RoboBrain and Dopamine use the current trajectory's real future endpoint.
FoREST rebuilds native causal memory and clears it between episodes.

Pearson/Spearman: per trajectory versus original PT then equal episode mean.
Endpoint: last ceil(2% * sampled-frame-count), only annotated goal-complete
episodes; not success classification accuracy. FPL: pinned RoboValue official
largest ZigZag descent onset, coefficient .2, native no-descent penalty,
normalized original frame duration. Applies to four regressive-failure
episodes. Repeat-only episodes are not treated as regressive failures.
Platform MAE: absolute PT and baseline-relative increment errors on explicitly
annotated repeated-action intervals, excluding idle/waiting. Applies to six
episodes. Equal episode aggregation, frame weighted within an episode.

All ten success labels are derived from accepted visual stage annotations'
goal-completion milestones, not PT==1. Two spray-bottle physical fall onsets
were independently reviewed from original 5FPS main-view frames and enlarged
neighborhoods, before model inference: episode30 frame762, episode31 frame768.
Resolution/uncertainty .2s, assistant-reviewed, not human-verified. Source
annotations place PT rollback at recovery onset; that is not used as failure
onset. Evidence and overlays remain remote under setup/event-review and events.

Remote root /cache/cvpzhr/setup/recover-five-validation-20261114
Outputs /cache/cvpzhr/evaluations/recover-five-validation-20261114
Finite detached queue, one process per idle GPU, allocator fraction .65 and
16GiB minimum free VRAM. Conservative cgroup resident/nonreclaimable guards.
Unknown failures stop that model; other models continue. No desktop automation.
Per-frame prefix resume validates input/weight/PT/frame hashes, never replaces
saved predictions. FoREST replay reconstructs memory without new language
predictions. New launch requires proof the old process and descendants exited.

Stages: prepare freeze -> decode and labels -> prepare finalize ->
review_evidence -> events -> controller. Do not repeat a completed freeze or
label generation. report can be safely rerun using CPU only.
