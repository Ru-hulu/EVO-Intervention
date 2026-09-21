# pi05 normalization statistics

pi05 normalizes proprioceptive state inputs and action targets using statistics computed over the training data. The statistics are stored alongside checkpoints under `assets/<asset_id>`.

For a fine-tuning run, either reuse the matching pi05 checkpoint statistics or compute fresh statistics:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
uv run scripts/compute_norm_stats.py --config-name pi05_droid_finetune
```

The DROID RLDS workflow uses `pi05_full_droid_finetune` and the LIBERO workflow uses `pi05_libero`.
