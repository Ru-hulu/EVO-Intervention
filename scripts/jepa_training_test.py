"""JEPA training utilities: resume equivalence, legacy checkpoints and persistent logs."""

import copy
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from evo_intervention.training import train_world_model as trainer
from scripts.jepa_piperx_test import tiny_qwen


@pytest.fixture
def training_setup(tmp_path, tiny_qwen, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    # Exercise stochastic predictor behavior as well as Python/NumPy RNG restoration.
    tiny_qwen.predictor.predictor_blocks[0].mlp.drop.p = 0.2
    samples = [{
        "frames": torch.full((3, 3, 64, 64), i * 30, dtype=torch.uint8),
        "actions": torch.full((2, 14), i / 10), "states": torch.full((3, 14), i / 10),
        "control_actions": torch.full((2, 1, 14), i / 10), "valid": torch.ones(3, dtype=torch.bool),
    } for i in range(6)]
    monkeypatch.setattr(trainer, "build_datasets", lambda cfg, contract: (samples, samples[:2]))
    monkeypatch.setattr(trainer.VJEPA2ACWorldModel, "from_config", lambda cfg: copy.deepcopy(tiny_qwen))
    compute_losses = trainer.compute_losses
    seen_batches = []

    def stochastic_loss(model, batch, cfg, state_transition=None):
        out = compute_losses(model, batch, cfg, state_transition)
        if model.predictor.training:
            seen_batches.append(batch["frames"][:, 0, 0, 0, 0].tolist())
            out["loss"] = out["loss"] * (1 + random.random() + np.random.random() + torch.rand(()))
        return out

    monkeypatch.setattr(trainer, "compute_losses", stochastic_loss)
    cfg = OmegaConf.create({
        "run_root_dir": str(tmp_path / "runs"), "run_id": "reference", "seed": 239,
        "world_model": {"encoder": {"type": "qwen3_5", "model_path": str(tmp_path)},
                        "predictor": {"action_dim": 14}},
        "contract": {"actions_per_latent_step": 1, "action_reduce": "last", "action_dim": 14, "state_dim": 14},
        "data": {"batch_size": 2, "num_workers": 0},
        "loss": {"rollout_steps": 2, "loss_exp": 1.0},
        "trainer": {
            "resume_from_checkpoint": None, "max_train_steps": 7, "warmup_steps": 2,
            "lr": 0.001, "start_lr": 0.0001, "final_lr": 0.0,
            "weight_decay": 0.04, "betas": [0.9, 0.999], "max_grad_norm": 1.0,
            "mixed_precision": "no", "log_interval": 1, "eval_interval": 2,
            "eval_batches": 1, "save_interval": 1,
        },
    })
    return cfg, seen_batches


@pytest.mark.parametrize("interrupt_step", [2, 3, 4])
def test_resume_matches_uninterrupted_training(training_setup, monkeypatch, interrupt_step):
    cfg, seen = training_setup
    trainer.main(cfg)
    reference_batches = seen.copy()
    reference = torch.load(f"{cfg.run_root_dir}/reference/checkpoints/final.pt", weights_only=True)
    cfg.run_id = "resumed"
    seen.clear()
    save = trainer.save_checkpoint

    def interrupt_after_save(path, accelerator, cfg, step, loader):
        save(path, accelerator, cfg, step, loader)
        if step == interrupt_step:
            raise RuntimeError("Simulated interruption")

    monkeypatch.setattr(trainer, "save_checkpoint", interrupt_after_save)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        trainer.main(cfg)
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    cfg.trainer.resume_from_checkpoint = f"{cfg.run_root_dir}/resumed/checkpoints/steps_{interrupt_step}"
    trainer.main(cfg)
    resumed = torch.load(f"{cfg.run_root_dir}/resumed/checkpoints/final.pt", weights_only=True)
    assert seen == reference_batches  # Mid-epoch, epoch boundary, and the next epoch.
    assert resumed["step"] == reference["step"] == 7
    for key in reference["predictor"]:
        torch.testing.assert_close(resumed["predictor"][key], reference["predictor"][key], rtol=0, atol=0)
    run = Path(cfg.run_root_dir) / cfg.run_id
    log = (run / "train.log").read_text()
    assert [int(step) for step in re.findall(r"step (\d+): train/loss=", log)] == list(range(1, 8))
    assert "Resumed" in log and "Training failed" in log and "val/loss=" in log
    state = json.loads((run / "checkpoints/final/trainer_state.json").read_text())
    assert state["step"] == 7
    scheduler = torch.load(run / "checkpoints/final/custom_checkpoint_0.pkl", weights_only=True)
    assert scheduler["last_epoch"] == 7
    assert (run / "checkpoints/final/random_states_0.pkl").is_file()


def test_legacy_resume_restores_optimizer_step_and_lr(training_setup):
    cfg, _ = training_setup
    trainer.main(cfg)
    checkpoint = Path(cfg.run_root_dir) / "reference/checkpoints/steps_2"
    legacy = checkpoint.parent / "steps_2.pt"
    torch.save({
        "predictor": load_file(checkpoint / "model.safetensors"),
        "optimizer": torch.load(checkpoint / "optimizer.bin", weights_only=True),
        "step": 2, "config": OmegaConf.to_container(cfg, resolve=True),
    }, legacy)
    cfg.run_id = "legacy"
    cfg.trainer.resume_from_checkpoint = str(legacy)
    trainer.main(cfg)
    run = Path(cfg.run_root_dir) / cfg.run_id
    state = torch.load(run / "checkpoints/final/optimizer.bin", weights_only=True)
    assert all(value["step"].item() == 7 for value in state["state"].values())
    assert state["param_groups"][0]["lr"] == 0
    log = (run / "train.log").read_text()
    assert "Legacy .pt resume" in log and "step 3:" in log and "step 2:" not in log
    # Step 3 must use the continued cosine schedule, rather than restarting warmup.
    expected_lr = cfg.trainer.lr * trainer.lr_lambda(cfg)(3)
    logged_lr = float(re.search(r"step 3:.*train/lr=([^\s]+)", log)[1])
    assert logged_lr == pytest.approx(expected_lr, rel=1e-3)


@pytest.mark.parametrize("invalid", ["weights_only", "incomplete", "changed_config"])
def test_resume_rejects_invalid_checkpoint_and_logs_error(training_setup, invalid):
    cfg, _ = training_setup
    path = Path(cfg.run_root_dir).parent / invalid
    if invalid == "weights_only":
        torch.save({"predictor": {}, "step": 2}, path)
        message = "no optimizer state"
    elif invalid == "incomplete":
        path.mkdir()
        message = "Incomplete training checkpoint"
    else:
        trainer.main(cfg)
        path = Path(cfg.run_root_dir) / "reference/checkpoints/steps_2"
        cfg.seed += 1
        message = "config mismatch in seed"
    cfg.run_id = "invalid"
    cfg.trainer.resume_from_checkpoint = str(path)
    with pytest.raises(ValueError, match=message):
        trainer.main(cfg)
    assert message in (Path(cfg.run_root_dir) / "invalid/train.log").read_text()
