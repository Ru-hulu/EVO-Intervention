Robometer five-task uniform history experiment, 2026-11-10

Each of pencil, flipcup, spray, pour and drawer starts independently from official
Robometer-4B revision beef63bc914c5c189329d49c6d712d96d632aa34. Existing data splits,
original PT, 5 FPS observations and final-frame inclusion are unchanged.

Only temporal sampling changes: sample eight times uniformly from the observed
5 FPS prefix including the first and current times, using numpy.linspace/rint.
When fewer than eight observations exist, append repetitions of the current time.
Keep seven sampled main-view slots plus current main/left/right (ten images).
The original input_contract name describes this spatial/token layout; the extra
temporal_sampling field prevents the new runtime from accepting old windows.
Official reference: helpers.py::linspace_subsample_frames and
pad_trajectory_to_max_frames_np at Robometer commit
352d160389daa964788de1ec933d1925f3a6de4f. check_sampling.py compares their exact AST
functions with 10001 online prefixes and verifies reset at episode boundaries.

The runtime and trainer were copied from the actual prior official-weight pencil
experiment, /cache/cvpzhr/setup/forest-robometer-pencil-official-e3-20261109/robometer.
The unchanged baseline_configs preserve the five prior configurations for review.
Changes to runtime/data.py and the added runtime/uniform.py apply sampling online
and enforce the cache contract. Model, loss, optimizer and training remain unchanged.

3 epochs, two GPUs (2,3), batch9/GPU, effective batch18, lr1e-4, 3% warmup+cosine,
fresh language LoRA rank16/alpha32/dropout.05 plus official progress head; seed42.
Save all three epochs before evaluation; evaluate only epoch3. MAE is frame-equal
in percentage points; Pearson is the equal mean of per-episode correlations, with
constant sequences counted explicitly as undefined. Plots use original dense PT
and unsmoothed 5 FPS predictions with hold-between-observation rendering.

build_cache.py rewrites every temporal sample and compares cached versus uncached
processor tensors at seven positions in every episode. RGB and independent BF16
per-frame pixels are immutable hardlinks to previous caches, avoiding duplicate
image preprocessing/storage. No previous cache, training output or label is edited.
Source manifests and output JSONLs are hashed; train/val disjointness is enforced.

Deploy this directory to /cache/cvpzhr/setup/robometer-five-uniform8-official-e3-20261110,
run configure.py and generate code-manifest.json (SHA256 of scripts/configs).
Use the existing robometer environment with PYTHONPATH=runtime:<pinned upstream>.
pipeline.py --prepare-only prepares and verifies all five caches without GPUs.
pipeline.py runs the strictly serial smoke -> train -> final validate -> verify ->
plot queue. The next task cannot start until all preceding outputs pass verification.
Smoke weights are isolated and never reused. GPU work waits for both assigned cards
to be idle. Failures stop the queue and preserve all artifacts; incomplete training
requires an explicit valid same-configuration checkpoint recovery audit.

Only experimental code/configuration belongs in Git. Data, generated manifests,
status/logs, images, weights, metrics and predictions stay on the A100 server.
