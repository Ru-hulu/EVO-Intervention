# pi05 training and evaluation

This repository retains the pi05 training, evaluation, and dataset-processing pipeline.

## Supported paths

- JAX and PyTorch training: `scripts/train.py` and `scripts/train_pytorch.py`
- Normalization statistics: `scripts/compute_norm_stats.py`
- Policy serving: `scripts/serve_policy.py`
- JAX-to-PyTorch checkpoint conversion: `examples/convert_jax_model_to_pi05_pytorch.py`
- DROID data conversion and evaluation: `examples/droid/`
- LIBERO data conversion and evaluation: `examples/libero/`

Available configurations are defined in `src/openpi/training/config.py` and are limited to `pi05_*` plus `debug_pi05`.

Install with:

```bash
uv sync
```

Run a basic training smoke test with:

```bash
uv run scripts/train.py debug_pi05
```
