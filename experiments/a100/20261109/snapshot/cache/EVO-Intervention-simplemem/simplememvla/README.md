# SimpleMem video memory + Robometer heads

The active reward-model entry point is `SimpleMemForRewardPrediction`: native
Qwen3.5 timestamped video history, with progress, success and preference heads.
It does not construct DiT, predict actions, require proprioception, or generate
sub-task text. The previous upstream action-model files remain as reference and
are not used by the reward-model forward path.

## Sources

- SimpleMemVLA: https://github.com/OpenBMB/SimpleMemVLA at
  `453d6ec293e63318bbc2bce545d0b9256ab1bc51`. Original model/configuration/DiT and
  messages/collator files are preserved; exports are extended for the reward model.
  MIT notice: `LICENSE`. Full clone: `/Users/dsy/MINT/SimpleMemVLA`.
- Robometer: https://github.com/robometer/robometer at
  `352d160389daa964788de1ec933d1925f3a6de4f`.
  See `ROBOMETER_SOURCE.md` for source files, attribution and adaptation differences.

## Model and input

```python
from transformers import AutoProcessor
from simplememvla.model import SimpleMemRewardConfig, SimpleMemForRewardPrediction
from simplememvla.data.reward import SimpleMemRewardCollator

path = "/path/to/Qwen3.5-4B"
processor = AutoProcessor.from_pretrained(path)
model = SimpleMemForRewardPrediction.from_backbone(
    path, config=SimpleMemRewardConfig(), dtype="bfloat16",
)
model.prepare_processor(processor)  # register <|pref_token|>; before optimizer creation
collator = SimpleMemRewardCollator(processor)

# features: see contract below. Move the collated tensors to the model's device.
batch = collator(features)
output = model(**batch)
output.loss.backward()

model.save_pretrained("/path/to/checkpoint")
processor.save_pretrained("/path/to/checkpoint")
# Reload with SimpleMemForRewardPrediction.from_pretrained(...), not AutoModel.
```

Each feature contains `instruction`, `videos`, `video_metadata` and optional
`images`. Supply one history video per trajectory, with one trajectory for
standalone scoring or two in A/B order for preference. `images` holds current
camera frames, in A/B order when paired. Metadata carries original sampled frame
indices and native fps, following upstream SimpleMem, so timestamps reflect real
time. The collator disables additional frame sampling. Existing
`data.messages.derive_video_sampling` and `variable_history_frames` provide the
upstream window/stride and even-frame rules; callers must select history up to
the current decision, never future frames. Configured history defaults are 60 s
at 2 fps; these values describe the intended sampler, not an automatic truncation
inside the model. Streaming inference is available separately as described below.

Targets are optional:

- `progress_targets`: [number of temporal patches], continuous [0,1], labels from
  each patch's last sampled frame; -100 skips supervision. Default is 10-bin C51
  cross entropy. Config also supports `progress_loss_type="l1"` or `"l2"`.
- `success_targets`: same layout, binary 0/1, -100 skips supervision; balanced BCE.
- `preference_targets`: scalar 1 when A is preferred, 0 when B is preferred; BCE
  on the comparison logit. Requires two trajectories and a registered pref token.

For paired samples, temporal labels concatenate A then B. B's representations
can see A in the joint prompt. For standalone trajectory scores use standalone
samples. Labels are never inserted into the prompt. Use batches with the same
supervision keys and do not mix preference-token and standalone samples.

Output `progress` is [batch,max_temporal_patches], `success_logits` has the same
shape, and `preference_logits` is [batch] for paired inputs. Apply sigmoid to
success/preference logits to get probabilities. `progress_mask` marks valid
patches; `video_indices` identifies A=0/B=1 and uses -1 for padding. Ignore padded
logits. Discrete mode additionally exposes `progress_logits`.

## Main-camera streaming

```python
from simplememvla.model import SimpleMemRewardStream

# For the verified numerical-equivalence path, load the backbone with
# dtype="float32", device_map="cuda", attn_implementation="sdpa" above.
model.eval()
stream = SimpleMemRewardStream(model)
# One unpadded sample, one main-camera video, no current images.
batch = collator([feature])
# One immutable key per temporal patch, normally a pair of sampled frames.
# Include content/preprocessing identity if frame indices can be reused.
output = stream.step(batch, patch_keys)
print(stream.stats)
stream.reset()  # between episodes, or after changing weights
```

Complete frame-pair appends encode only new visual patches and run only the new
language tokens, reusing full-attention KV and linear-attention recurrent/conv
states. Repeated inputs do no backbone work. Prompt/position changes, a sliding
window, or replacement of the last odd padded pair rebuild language state;
unchanged visual features remain reusable. Only live-window visual features are
kept. KV grows with an expanding history; this is not an unlimited-memory cache.
For incremental CPU input preparation, use `SimpleMemRewardPreprocessor` below.
The original collator remains the full-history reference and training path.
Streaming returns progress/success for all current patches. Paired preference
scoring remains available through the regular model forward.

```python
from simplememvla.data.streaming import SimpleMemRewardPreprocessor

inputs = SimpleMemRewardPreprocessor(processor, instruction, fps=2.0)
# Supply only NEW complete pairs, uint8 RGB [2,H,W,3]. For three cameras,
# concatenate their new synchronized pairs horizontally before this call.
# Frames must already have the fixed target resolution, a multiple of 32
# for the standard Qwen3.5 processor. Resize only new frames if needed.
batch = inputs.append(new_rgb_frames, source_frame_indices)
output = stream.step(batch, inputs.patch_keys)
# At the next episode:
inputs.reset()
stream.reset()
```

This CPU cache normalizes/patchifies only new pairs and tokenizes only the new
timestamp spans. It keeps old pixels in a growing CPU buffer and old prompt/video
tokens in memory. Full-history tensor views are returned for compatibility with
the existing model stream; old pixels are not uploaded or encoded again. Buffer
growth occasionally copies existing pixels. Inputs must be append-only with
increasing source indices and fixed processor settings; incomplete pairs,
replayed indices or changed resolution are rejected. Construct a new preprocessor
for a different task/fps, and reset the model stream too. No sliding-window or
odd-frame replacement API is provided by this CPU adapter.

In a CPU-only 130-second three-view test, all five model input tensors matched
the native full-history collator exactly at the checked prefixes including 120
and 130 seconds. At 111–120 seconds, median CPU preparation was 2.67 ms; the
largest buffer-growth update across the run was 97.50 ms. The cache holds about
675 MiB of pixel data at 120 seconds (720 MiB buffer capacity). Updated full-size
GPU latency, VRAM and output comparisons are pending available GPU resources;
the earlier GPU measurements above do not include this CPU optimization.

For Transformers 5.3, this adapter enables GDN state reuse for multi-token
appends (the native path only resumes single-token decoding). FP32 uses the
PyTorch GDN kernel and CUDA memory-efficient SDPA with explicit KV-head expansion.
These per-model adaptations also apply to subsequent full forwards. Construct
the stream after placing/casting the model, and keep its weights/device/dtype fixed.

On 4090C with the pretrained Qwen3.5-4B backbone, FP32 streaming matched full
recomputation within 2.2e-5 absolute probability error over the tested 10–70 s
histories. BF16/FP16 had materially larger numerical differences and should not
be assumed equivalent. Heads were randomly initialized: this validates execution,
not trained task accuracy. See the local experiment report under
`outputs/experiments/simplemem-stream/2026-11-06_4090C/` for timings and limits.

## Validation and scope

Run `uv run --no-sync --with scipy==1.14.1 python -m unittest tests.test_simplemem_reward`.
The temporary SciPy overlay bypasses a local macOS dylib issue without changing
project dependencies. Tests use a real tiny Qwen3.5 vision/language backbone,
including video forward/backward, future-patch causality, variable-length
batches, masking, preference placement and checkpoint roundtrip. Collator tests
use deterministic processor outputs. Streaming tests cover append, repeat,
replacement, window slide, prompt changes and reset. The real pretrained
processor/backbone was additionally tested on 4090C; GPU training was not.

The project remains on torch 2.10 / Transformers 5.3. Upstream SimpleMemVLA declares
torch 2.4.1 / Transformers >=5.11. No lockfile update, pretrained head
transfer, dataset wiring, training launcher or simulator integration is included.

## ProPriMem training (simplemem branch)

The Huawei stage-1 baseline reads the existing cloud progress CSVs directly:

```bash
cd /cache/EVO-Intervention-simplemem
/cache/EVO-Intervention/.venv/bin/python -m simplememvla.training.prepare_huawei
bash simplememvla/training/run_huawei.sh
# Plumbing test, not a quality or full-context memory benchmark:
bash simplememvla/training/run_huawei.sh --max-steps 1 --accumulate 1 \
  --max-frames 4 --eval-samples 1 --out outputs/training/smoke
```

`prepare_huawei` imports the ordinary cap50 dataset's frozen sample lists and
recovery label metadata. It preserves original-video splits, excluded O03
`episode_001342` (mismatched camera lengths), and excluded recovery `file-019`.
The baseline includes ordinary successful trajectories and insert-screw-recover
cycles. It does not include the separate S01 failure/anomaly dataset or normal
insert-screw extras. Success and preference supervision are disabled explicitly.

Ordinary cloud labels use adjacent-frame L2 distances of mean post-merger visual
features, cumulatively summed over the whole video and normalized to [0, 1].
The existing generator explicitly sets `annotations_used=False`: these labels
are not normalized separately at annotated subtask boundaries. Recovery labels
use the original `episode_to_progress` algorithm: subtask delta contributions
are scaled to the same task ID's last execution, invalidated task contributions
are subtracted at retry boundaries, and each cycle is normalized independently.
The importer preserves those declines and exact values, verifying CSV checksums
and contiguous source-frame indices. It does not regenerate labels.

Recovery samples select only their cycle's source-frame interval, preserving
original frame indices for CSV lookup and timestamps. Old absolute media paths
are mapped to the current task-relative camera paths. CSV checksums and video
sizes are checked when reading training samples; videos are not rehashed or
fully decoded by the importer. Existing camera alignment is inherited from
Huawei's preparation, not reconstructed from timestamps.

The default now updates the VLM and progress head (`huawei_full.json`), using
FP32 master weights, CUDA BF16 autocast and gradient checkpointing. AdamW uses
1e-5 for the backbone and 3e-4 for the progress head, with batch size one and
gradient accumulation. `--config simplememvla/training/configs/huawei_heads.json`
selects the earlier frozen-VLM baseline. Unsupervised
success/preference heads are frozen and excluded from optimizer decay. All
existing views are resized to 320×256 and mosaicked in camera order; two-view
samples remain two-view. Each two-frame temporal patch receives the progress
of its last source frame. Training does not use inference caches.

Full history is sampled at up to 2 fps, capped at 240 frames by uniform
subsampling while retaining endpoints. Long videos have reduced temporal
density, which can miss short recovery events. This is a sampling choice,
independent of the original per-frame labels. `--mode full` enables FP32
backbone updates with gradient checkpointing, without DDP/ZeRO/LoRA; a 4B AdamW
run needs roughly 64 GiB for model/gradient/optimizer state before activations.
Full-size backward and training quality require separate evaluation.

Checkpoints retain optimizer, RNG, sampling settings, dataset digest and epoch
position. Heads-only checkpoints require the same base model. Resume with
`--resume outputs/training/proprimem_huawei_vlm_progress/last.pt`; changing data or
key sampling settings is rejected. Validation reports micro-averaged progress
MAE/loss; `eval_samples=0` evaluates all validation samples. This is not a
balanced task metric or memory/cycle VOC benchmark.

`simplememvla.training.prepare` remains an optional audit of raw Hub stage
annotations. Its event-interpolated labels and missing-annotation counts do
not describe Huawei's existing visual-delta labels and are not the default
training input. Use `prepare_huawei` for the cloud baseline.

The reward training path currently has no subtask/stage prediction supervision.
It does not reproduce upstream SimpleMemVLA's generated-subtask text channel.
Existing streaming benchmarks therefore do not establish end-to-end latency
for a monitor that also predicts stages or generates subtask descriptions.

Current-action label normalization is available through `prepare_huawei`:

```bash
/cache/EVO-Intervention/.venv/bin/python -m simplememvla.training.prepare_huawei \
  --out outputs/proprimem-huawei-stage1-current-action
```

Records now include `current_actions` or null when boundaries are unavailable.
Intervals use original, zero-based, half-open source-frame ranges. Existing
current-action annotations retain their atomic intervals (including separate
pick/place and right/left movements). Cumulative completed states are converted
to action phases ending at the next completion boundary; the last completed
state becomes terminal, not an additional action. Such phases are explicitly
marked `inferred_from_completion_boundaries`: they do not identify physical
motion onset or distinguish pauses. Recovery retains every retry and converts
inclusive source boundaries to exclusive ends. Stage IDs are local to each
record's vocabulary, not global classifier labels. The existing CSV progress,
frozen splits and missing-boundary examples are preserved.

This produces normalized supervision metadata; stage loss, a shared vocabulary
or task-conditioned classifier, and online stage prediction are not yet wired
into the reward trainer. Progress CSVs have not been regenerated under a new
unified stage-based progress policy.
