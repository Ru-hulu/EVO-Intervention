Original Qwen3.5-4B for visual progress labels

The official `Qwen/Qwen3.5-4B` snapshot is pinned to
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` in
`qwen35-download-catalog.json`. The catalog includes all 14 repository files,
9,342,907,469 bytes, with the official Git blob SHA1 or LFS SHA256.

Run on A100 from the repository root:

```bash
python -m experiments.proprimem_visual_pt_20261113.download_qwen35 \
  --state-dir /cache/cvpzhr/setup/qwen35-original-download-20261113 \
  --workers 4
```

All model files download directly on A100 to
`/cache/cvpzhr/models/Qwen3.5-4B/`. The downloader resumes partial files,
holds an exclusive job lock, verifies each official hash before publishing a
file, and checks that the weight index references verified shards. Large files
prefer the author's faster ModelScope CDN; every file must match the pinned
official Hugging Face hash. No Hugging Face token is sent to either mirror.

Completion and per-file hashes are recorded in `status.json` and
`verification.json` in the state directory. This command does not remove time
labels, generate new labels, or start training or evaluation. Model weights,
datasets, labels, and runtime status files are not stored in Git.
