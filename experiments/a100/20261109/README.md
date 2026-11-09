# A100 experiment source snapshots — 2026-11-09

This directory preserves the exact source versions and configurations read from the A100 host on 2026-11-09. It is a source archive, not a new training run. Existing branch source trees are not overwritten or merged across model variants.

## Layout and branch ownership

- `externmem-histoken-streamforest`: original Forest, cap-50 and S01 experiments, anomaly-task training, pencil/spray warm-start reruns, and flipcup cache preparation.
- `externmem-histoken-crossAtten`: external-memory cross-attention experiments and mixed-screw cache/training variants.
- `externmem-histoken-crossAtten-linearAtten`: linear-attention experiments and the screw-pair preparation based on that checkpoint.
- `internmem-kvcache-VST`: VST QKV experiment source and configuration.
- `main`: Robometer, RoboBrain2.5, cross-model evaluations, shared data-preparation code, legacy Qwen/SimpleMem source snapshots, and downstream baseline experiment scripts/patches.
- Other existing branches remain unchanged: no distinct A100 experiment snapshot was identified for them in this collection.

The same `experiments/a100/20261109/` path exists on each updated branch. `manifest.json` lists only the files in that branch; `branch-map.json` locates all captured files across branches.

## Reproducing an experiment

`snapshot/cache/...` mirrors the original `/cache/...` paths. Use the exact experiment's own `code/` directory and configuration, rather than mixing it with a different run or the branch-root package. Configurations retain original absolute data, model, checkpoint and output paths, split references, seeds and hyperparameters. These paths are references; their target files are not included.

On another host, restore the required datasets from Hugging Face and model/checkpoint dependencies separately, install the recorded requirements in an isolated environment, and adapt the path references before running. Read the corresponding `pipeline.py`, launch script and configuration: some pipelines wait for other jobs or begin training automatically. Archiving these files did not execute them or change A100 jobs. Dependency versions and runtime services are not bundled.

## Data policy

No videos, image observations, per-frame labels/deltas, dataset shards, feature caches, model weights, checkpoints, predictions, metrics files, training logs or curve images are added. Only source, experiment configuration, dependency/license information and source provenance are included. Python packages named `datasets` contain preprocessing code, not dataset payloads. Existing historical repository content is not removed by this source-only addition.

## External repositories

On `main`, `external-repositories.json` pins inspected upstream repositories. `external-patches/` contains only local source modifications and new source files, with the original repository and base commit in each `SOURCE.json`. Apply `changes.patch` to that exact upstream commit and copy `untracked/` files using their relative paths. Upstream repositories themselves are not modified or pushed by this synchronization. A changed submodule revision is recorded separately through the inspected nested repository commit.

The remote `/cache/EVO-Intervention-simplemem` branch used a local Git bundle as its origin; its current source is preserved under the `main` snapshot with that original directory name, rather than claiming it was an existing GitHub branch.
