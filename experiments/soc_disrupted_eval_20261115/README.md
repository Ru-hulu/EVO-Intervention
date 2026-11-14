# SOC observation disruption evaluation

Code preparation only. No inference starts on import, preparation, CPU verification,
or report generation. GPU canaries and formal evaluation require the explicit
controller command below. No other controller, training process, or dataset is modified.

## Fixed inputs and weights

- Models: ProcVLM-2B, Robometer-4B, FoREST, with native visual memory enabled.
- Every model uses its matching suite's final **35-training-trajectory, 1-epoch**
  fine-tuned checkpoint. Nine checkpoint identities and loaded tensor digests are audited.
- Frozen selection: State 22, Order 32, Count 30 validation trajectories, two per
  task. Training and validation selections must be disjoint.
- 5 FPS; static and rewind evaluated separately, two insertions per trajectory,
  each lasting 5–15 observation steps; rewind adds an equally long forward replay.
- Seeds 42/43/44 are shared across all models: 504 curves and 105,286 predictions
  per model, 1,512 curves and 315,858 predictions overall.
- Cached PNGs, original labels, splits, and perturbation maps are read-only.
  The existing missing-view black placeholder policy is preserved.

ProcVLM receives the latest eight **presented** main-view images, left padded with
the first image, and its SOC35 progress-only text prompt. It has no previous CoT
input. Robometer uniformly samples eight steps from the **presented prefix**,
right padding early histories with the current image, and uses seven main views
plus the current three views. FoREST uses its original CLIP/max_depth memory and
capacity. Its memory position increases with playback, in native-frame units;
source frame identity can decrease during rewinds. No model receives PT, event
boundaries, or future frames. Every new trajectory/condition resets memory.

## Metrics

Pearson and Spearman compare the full presented prediction curve with the mapped
PT curve. Spearman uses average ranks for ties. Scores are computed per trajectory
then averaged equally within each task/suite. Each seed is reported separately,
with mean and sample standard deviation across the three seed-level means.

For a rewind event, pair the original forward visit, reverse visit, and forward
replay of each interior source frame. With predictions `p1,p2,p3`, compute
`(|p3-p1| + |p2-p1| + |p3-p2|) / 3`. Average all timestamp triplets from the two
events within a trajectory. Turning endpoints have two visits and are excluded.
Overlapping source intervals are paired independently per insertion, always using
the original forward visit as `p1`. Every triplet and its three indices/predictions
is retained. Lower is more consistent; a constant predictor may score zero.

Static MAE is the mean absolute deviation of predictions from **their own segment
mean**, using only the newly inserted static frames (not the original anchor).
The trajectory score equally averages the two segment MAEs. This is not error
against PT. Lower is more stable; a constant predictor may also score zero.

Invalid responses remain raw/null. No clipping, interpolation, carry-forward,
smoothing, or per-curve normalization is applied. Any invalid prediction makes
full-curve VOC undefined. Each disruption metric requires all of its participating
predictions; invalid frames outside that region do not invalidate that metric.
Complete suites report defined-trajectory means with explicit defined/undefined
denominators; incomplete suites do not publish partial-subset means. No cross-seed
mean is published unless all three seeds have complete coverage and defined means.

## Entrypoints

Run from the isolated code snapshot at
`/cache/cvpzhr/setup/soc-disrupted-eval-20261115/code`.

CPU preparation/verification, safe while other GPU jobs run:

```sh
CUDA_VISIBLE_DEVICES= /cache/cvpzhr/envs/forest-legacy-20261114/bin/python -m pytest experiments/soc_disrupted_eval_20261115/test_disrupted.py -q -p no:cacheprovider
CUDA_VISIBLE_DEVICES= /cache/cvpzhr/envs/forest-legacy-20261114/bin/python -m experiments.soc_disrupted_eval_20261115.verify_cpu
CUDA_VISIBLE_DEVICES= /cache/cvpzhr/envs/forest-legacy-20261114/bin/python -m experiments.soc_disrupted_eval_20261115.prepare --prepare-only
```

**Not executed during code preparation.** A later authorized launch uses:

```sh
nohup /cache/cvpzhr/envs/forest-legacy-20261114/bin/python -m experiments.soc_disrupted_eval_20261115.controller --start-evaluation > /cache/cvpzhr/setup/soc-disrupted-eval-20261115/controller.log 2>&1 < /dev/null &
```

The isolated controller queues nine complete model/suite tasks, each occupying
exactly one GPU. Each process first runs its suite-specific real GPU canary,
evaluates complete static and rewind streams and compares continuous
inference with resumes inside both inserted events (maximum difference 0.002,
matching invalid-answer locations). Canary predictions never enter formal results.
After that task's canary passes, the same process evaluates all suite trajectories
across the six conditions. It does not split a model/suite across multiple GPUs.
The model resets between versions; other queued tasks advance as GPUs become idle.

Admission requires an idle GPU with no NVML compute context, 60 seconds of stable
resources, and at least 60 seconds between launches. Each GPU has at most one
worker, batch 1, allocator fraction 0.70, and at least 16 GiB headroom. Container
resident/shared memory, non-reclaimable usage, OOM, temperature, and disk guards
remain active. Unknown contexts are never assumed to belong to this task.

Every worker requires a live controller ticket and per-job lock. It saves a prefix
after every frame. A safety pause preserves that prefix; only safety pauses may
retry, with at most three launches/job/hour. Unknown failures stop new launches
while healthy workers finish. The controller never signals other processes.
To pause this queue cooperatively, create `setup/pause-all`; remove it to allow
safe resume. An exited controller requires explicit `--resume`, confirmation that
all old worker identities have exited, and passed unchanged-input checks.
FoREST reconstructs prefix vision memory without rerunning prefix language calls.

CPU report, callable at any time without starting inference:

```sh
CUDA_VISIBLE_DEVICES= /cache/cvpzhr/envs/forest-legacy-20261114/bin/python -m experiments.soc_disrupted_eval_20261115.report
```

Outputs are under `/cache/cvpzhr/evaluations/soc-disrupted-eval-20261115`:
per-condition/model `episodes/` files, `report.json` with every event/triplet,
`trajectory-metrics.csv`, and `comparison.csv`. The report rechecks input, label,
source, code, and checkpoint hashes, initialization audits, GPU canary identities,
frame maps, raw/prediction lengths, and recomputed metrics. No result is fabricated
for missing or invalid predictions.
