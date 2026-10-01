# EVO-Intervention

A minimal Qwen3.5-2B VLA training base, pruned from StarVLA (`starVLA` stable branch, commit `3422b9f`).

**Status: never run.** Every Python file parses and all internal imports resolve, and the requirements resolve for Linux / Python 3.10. Nothing has been installed or trained.

## How it works

```
LeRobot dataset ──> sample {image, lang, action, state?} ──> framework.forward() ──> {"action_loss"} ──> trainer
                                                             framework.predict_action()  (inference)
```

| Piece | Where |
|---|---|
| Frameworks (one file each) | `evo_intervention/model/framework/VLM4A/` |
| Qwen3.5 wrapper | `evo_intervention/model/modules/vlm/QWen3_5.py` |
| Action heads | `evo_intervention/model/modules/action_model/` |
| Sample packing | `_pack_sample` in `evo_intervention/dataloader/gr00t_lerobot/datasets.py` |
| Robot and dataset definitions | `data_config.py`, `mixtures.py` in `evo_intervention/dataloader/gr00t_lerobot/` |
| Training loop | `evo_intervention/training/train.py` |
| Config templates | `evo_intervention/config/training/` |
| Policy server | `deployment/model_server/` |

Frameworks, picked by `framework.name` in the YAML:

| Name | Action head |
|---|---|
| `QwenOFT` | MLP regression (simplest) |
| `QwenGR00T` | Flow-matching DiT on the last hidden layer |
| `QwenPI`, `QwenPI_v3` | Flow-matching DiT reading every VLM layer |
| `QwenFast` | Discrete FAST tokens; needs a checkpoint with action tokens added |

`examples/` is empty (your train and eval scripts go there). `tools/` holds the boundary annotator and corpus viewer.

## Setup

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
uv sync
```

The project is locked to Python 3.10, PyTorch 2.10.0 and the official CUDA 12.8 wheel index.

## First baseline run

Do this before adding anything, so there is a working reference.

1. Download `Qwen/Qwen3.5-2B` to `playground/Pretrained_models/Qwen3.5-2B`.
2. Register your robot in `data_config.py` and `mixtures.py`.
3. Copy `libero.yaml` and set `datasets.vla_data.data_root_dir`, `data_mix`, `action_dim`, `state_dim`, `wandb_entity`, `wandb_project`, `run_id`.
4. Set `trainer.freeze_modules: qwen_vl_interface` if the VLM should stay frozen. It is empty in the templates, which trains the VLM.
5. Run `accelerate launch evo_intervention/training/train.py --config_yaml <your.yaml>`.

## Known gaps

- **Intervention modules are not connected.** `model/modules/intervention/` is the pi05 code, unchanged. The window assumes image tokens followed by prompt tokens and layers 1–17; Qwen3.5-2B interleaves them and has 24 layers.
- **`QwenFast` token tool is unverified.** `modules/vlm/tools/add_qwen_special_tokens/` defaults to Qwen2.5.
- **`pyarrow` is `>=15`** instead of StarVLA's `==14.0.1`, so `flash-linear-attention` can install. Parquet loading is unchecked.
- **`tests/intervention_dataset_test.py`** imports a module that no longer exists.
