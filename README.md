# EVO-Intervention

A robot JEPA world-model training project. The current focus is a frozen Qwen3.5-2B vision tower
and an action-conditioned Transformer that predicts future visual latents.

The VLA modules inherited from StarVLA remain as reference code. Use the JEPA entry point below
for the PiperX dataset; it trains the latent predictor.

## How it works

```
LeRobot video windows ──> frozen Qwen vision tower ──> visual tokens
                                                          │
                     joint states + absolute actions ──> JEPA causal Transformer
                                                          │
                                                 future visual tokens
```

## Preparation on 4090

Prerequisites: an NVIDIA driver, CUDA Toolkit 12.8 and `uv`. Run from the project directory;
`uv sync` creates the Python 3.10 `.venv` and installs the locked dependencies.

```bash
cd ~/code/EVO-Intervention
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
uv sync --extra dev
nvidia-smi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"

# Keep downloads, caches, logs and checkpoints on the data disk.
mkdir -p ~/data/zhaobo/{datasets,models,cache/huggingface,checkpoints/EVO-Intervention}
export HF_HOME=~/data/zhaobo/cache/huggingface
# Optional mirror: export HF_ENDPOINT=https://hf-mirror.com
.venv/bin/hf download Takizz/piperx-bolt-intentional-drop-reinsert-22ep-20261001 \
  --repo-type dataset \
  --local-dir ~/data/zhaobo/datasets/piperx-bolt-intentional-drop-reinsert-22ep-20261001
.venv/bin/hf download Qwen/Qwen3.5-2B \
  --local-dir ~/data/zhaobo/models/Qwen3.5-2B
```

If the downloads are already complete, skip them. Check `data.root`, `world_model.encoder.model_path`
and `run_root_dir` in `evo_intervention/config/world_model/qwen3_5_piperx.yaml` against your local paths.
The current Qwen encoder config does not require V-JEPA encoder weights.

## PiperX JEPA training

The config is `evo_intervention/config/world_model/qwen3_5_piperx.yaml`.
It uses the downloaded `Takizz/piperx-bolt-intentional-drop-reinsert-22ep-20261001` dataset and
`/home/zhaobo/data/zhaobo/models/Qwen3.5-2B`. Checkpoints go to
`/home/zhaobo/data/zhaobo/checkpoints/EVO-Intervention/qwen35_piperx_jepa`.

- Only the Qwen vision tower and its native merger are loaded and frozen. Language weights are not loaded.
- A 256px frame produces an 8x8 grid of 2048-dimensional merged visual tokens.
- The randomly initialized JEPA predictor has 24 layers, width 1024 and 16 attention heads.
  Only its parameters are optimized, including its action/state and latent input/output projections.
- The source reads native LeRobot v3 parquet/video metadata, including per-camera file indices and timestamps.
  `meta/modality.json` and VLA task prompts are unnecessary.
- The selected camera is `observation.images.right_environment_1`. Change `data.camera_key` to select a wrist camera.
- Five frames are sampled every six control steps: 30 Hz recordings become 5 Hz latent frames, with four
  prediction steps spanning 0.8 seconds. Windows stay within an episode; 20 episodes train and 2 validate.
- State/action vectors preserve all 14 source dimensions: left joints 1–6, left gripper, right joints 1–6,
  right gripper. Both use native dataset min/max normalization to [-1, 1]. Absolute actions reduce with `last`.
- The loss is teacher-forced next-latent L1 plus four-step autoregressive rollout L1. Predicted visual latents
  feed back into rollout; recorded states provide state conditioning during training. Targets have no gradients.

Run the basic regression checks, then start training. Logs and checkpoints are saved to the data disk:

```bash
cd ~/code/EVO-Intervention
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest scripts/jepa_piperx_test.py scripts/jepa_training_test.py -q
HF_HUB_OFFLINE=1 .venv/bin/python -u -m evo_intervention.training.train_world_model \
  --config_yaml evo_intervention/config/world_model/qwen3_5_piperx.yaml
```

The default is batch size 2 with two data workers and BF16 computation. `train.log` in the run directory
contains timestamped training/validation metrics, checkpoint messages and errors, and is appended when resuming.
Checkpoint directories (`checkpoints/steps_1000`, etc., and `checkpoints/final`) contain the predictor,
optimizer, scheduler, RNG/scaler state, config and training position. The frozen encoder is not saved.
`checkpoints/final.pt` remains a predictor-only export for inference or weights-only initialization.

```bash
tail -n 50 -f ~/data/zhaobo/checkpoints/EVO-Intervention/qwen35_piperx_jepa/train.log

# Continue from a completed checkpoint; max_train_steps is the total target, not extra steps.
HF_HUB_OFFLINE=1 .venv/bin/python -u -m evo_intervention.training.train_world_model \
  --config_yaml evo_intervention/config/world_model/qwen3_5_piperx.yaml \
  trainer.resume_from_checkpoint=/home/zhaobo/data/zhaobo/checkpoints/EVO-Intervention/qwen35_piperx_jepa/checkpoints/steps_1000
```

Keep the original model/data, batch size, process count and learning-rate schedule when resuming.
The next batch is reconstructed from the saved step and deterministic epoch shuffle.
Old interval checkpoints such as `steps_1000.pt` also work with `trainer.resume_from_checkpoint`;
they restore predictor/optimizer/step and reconstruct the scheduler, but lack RNG/sampler/scaler state,
so the continuation cannot exactly reproduce the interrupted run. Old `final.pt` exports lack optimizer
state; use `world_model.checkpoint.predictor_path=<file>` for a new run initialized from those weights.
An already running process uses the code it loaded at launch; these utilities apply after launching the updated entry point.

Verified on the remote RTX 4090 with Python 3.10 / PyTorch 2.10.0+cu128 / Transformers 5.3.0:
JEPA regression tests, real episode boundary samples, and one full train/validation/checkpoint smoke step.
The 20/2 episode split produces 17,911 training windows and 1,807 validation windows.
Training metrics print every 10 steps, validation runs every 500, and checkpoints save every 1,000.

`evo_intervention/config/world_model/vjepa2_ac_vitg.yaml` retains the existing V-JEPA 2-AC encoder/NPZ setup.
The Qwen config uses its own frozen visual feature space and initializes the predictor from scratch.

## VLA reference modules

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

## VLA reference baseline

Optional reference for VLA experiments.

1. Download `Qwen/Qwen3.5-2B` to `playground/Pretrained_models/Qwen3.5-2B`.
2. Register your robot in `data_config.py` and `mixtures.py`.
3. Copy `libero.yaml` and set `datasets.vla_data.data_root_dir`, `data_mix`, `action_dim`, `state_dim`, `wandb_entity`, `wandb_project`, `run_id`.
4. Set `trainer.freeze_modules: qwen_vl_interface` if the VLM should stay frozen. It is empty in the templates, which trains the VLM.
5. Run `accelerate launch evo_intervention/training/train.py --config_yaml <your.yaml>`.

## Known gaps

- **Intervention modules are not connected.** `model/modules/intervention/` is the pi05 code, unchanged. The window assumes image tokens followed by prompt tokens and layers 1–17; Qwen3.5-2B interleaves them and has 24 layers.
- **`QwenFast` token tool is unverified.** `modules/vlm/tools/add_qwen_special_tokens/` defaults to Qwen2.5.
- **`pyarrow` is `>=15`** instead of StarVLA's `==14.0.1`, so `flash-linear-attention` can install.
- **`tests/intervention_dataset_test.py`** imports a module that no longer exists.
