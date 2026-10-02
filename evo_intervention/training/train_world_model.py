"""Train the JEPA AC predictor on robot trajectory windows with L_teacher-forcing + L_rollout.

The encoder stays frozen; only the predictor is optimized (random init, or the official AC
predictor when `world_model.checkpoint.predictor_path` is set). Plain Accelerate/DDP, no DeepSpeed.

    accelerate launch evo_intervention/training/train_world_model.py \
        --config_yaml evo_intervention/config/world_model/qwen3_5_piperx.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from evo_intervention.dataloader.world_model_windows import (
    LeRobotV3EpisodeSource, NpzEpisodeSource, TrajectoryWindowDataset, split_episodes,
)
from evo_intervention.model.modules.world_model import (
    LatentStepContract,
    VJEPA2ACWorldModel,
    build_state_transition,
    teacher_forcing_rollout_loss,
)


def build_source(cfg):
    if cfg.data.source == "npz":
        return NpzEpisodeSource(cfg.data.root)
    if cfg.data.source == "lerobot_v3":
        return LeRobotV3EpisodeSource(
            cfg.data.root, camera_key=cfg.data.get("camera_key"), camera_keys=cfg.data.get("camera_keys")
        )
    raise ValueError(f"Unknown data source '{cfg.data.source}'")


def build_datasets(cfg, contract):
    source = build_source(cfg)
    if cfg.data.window_frames < cfg.loss.rollout_steps + 1:
        raise ValueError(
            f"window_frames={cfg.data.window_frames} is too short for rollout_steps={cfg.loss.rollout_steps}"
        )
    train_eps, val_eps = split_episodes(len(source), cfg.data.val_fraction, cfg.seed)
    # Rollout loss must cover every configured step, so windows need at least rollout_steps + 1 frames.
    min_frames = cfg.data.min_frames or cfg.data.window_frames
    min_frames = max(min_frames, cfg.loss.rollout_steps + 1)
    kwargs = dict(source=source, contract=contract, window_frames=cfg.data.window_frames, min_frames=min_frames)
    return TrajectoryWindowDataset(episodes=train_eps, **kwargs), TrajectoryWindowDataset(episodes=val_eps, **kwargs)


def lr_lambda(cfg):
    t = cfg.trainer

    def fn(step):
        if step < t.warmup_steps:
            return (t.start_lr + (t.lr - t.start_lr) * step / max(1, t.warmup_steps)) / t.lr
        progress = min(1.0, (step - t.warmup_steps) / max(1, t.max_train_steps - t.warmup_steps))
        return (t.final_lr + 0.5 * (t.lr - t.final_lr) * (1 + math.cos(math.pi * progress))) / t.lr

    return fn


def compute_losses(world_model, batch, cfg, state_transition=None):
    with torch.no_grad():
        latents = world_model.encode(world_model.preprocess(batch["frames"]))
    return teacher_forcing_rollout_loss(
        world_model,
        latents,
        batch["actions"],
        batch["states"],
        rollout_steps=cfg.loss.rollout_steps,
        valid=batch["valid"],
        loss_exp=cfg.loss.loss_exp,
        state_transition=state_transition,
        transition_actions=batch["control_actions"],
        truncate_every=cfg.loss.get("truncate_every"),
    )


@torch.no_grad()
def evaluate(world_model, loader, cfg, accelerator, state_transition=None):
    """Exact masked means over up to `eval_batches` batches per process: error sums and valid
    counts are accumulated across batches and processes, then divided once.

    The distributed sampler may repeat a few samples to even out the last batch across processes.
    """
    world_model.predictor.eval()
    totals = {}
    for i, batch in enumerate(loader):
        if i >= cfg.trainer.eval_batches:
            break
        with accelerator.autocast():
            out = compute_losses(world_model, batch, cfg, state_transition)
        for k, (err, count) in out["sums"].items():
            prev_err, prev_count = totals.get(k, (0, 0))
            totals[k] = (prev_err + err.float(), prev_count + count.float())
    world_model.predictor.train()
    if not totals:
        raise RuntimeError("Validation produced no batches")
    reduced = {
        k: (accelerator.reduce(err, reduction="sum"), accelerator.reduce(count, reduction="sum"))
        for k, (err, count) in totals.items()
    }
    tf_err, tf_count = reduced["teacher_forcing"]
    step_err, step_count = reduced["rollout_per_step"]
    per_step = step_err / step_count.clamp_min(1.0)
    metrics = {
        "teacher_forcing": (tf_err / tf_count.clamp_min(1.0)).item(),
        "rollout": (step_err.sum() / step_count.sum().clamp_min(1.0)).item(),
        "rollout_final": per_step[-1].item(),
    }
    metrics["loss"] = metrics["teacher_forcing"] + metrics["rollout"]
    metrics.update({f"rollout_per_step/{j + 1}": x.item() for j, x in enumerate(per_step)})
    return metrics


def save_checkpoint(path: Path, accelerator, cfg, step, train_loader):
    """Save all prepared training state; write the completion marker last."""
    marker = path / "trainer_state.json"
    if accelerator.is_main_process:
        path.mkdir(parents=True, exist_ok=True)
        marker.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(path))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        OmegaConf.save(cfg, path / "config.yaml")
        marker.write_text(json.dumps({
            "step": step, "num_processes": accelerator.num_processes,
            "batches_per_epoch": len(train_loader),
        }), encoding="utf-8")
    accelerator.wait_for_everyone()


def load_checkpoint(path, world_model, accelerator, cfg, optimizer, scheduler, train_loader, logger):
    path = Path(path).expanduser()
    if path.is_dir():
        marker = path / "trainer_state.json"
        if not marker.is_file():
            raise ValueError(f"Incomplete training checkpoint: {path} (missing trainer_state.json)")
        state = json.loads(marker.read_text(encoding="utf-8"))
        if (state["num_processes"], state["batches_per_epoch"]) != (
            accelerator.num_processes, len(train_loader)
        ):
            raise ValueError("Resume requires the same process count and training batches per epoch")
        previous_cfg = OmegaConf.load(path / "config.yaml")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "optimizer" not in state:
            raise ValueError(
                "This .pt checkpoint has no optimizer state. For a weights-only start, use "
                "world_model.checkpoint.predictor_path instead of trainer.resume_from_checkpoint."
            )
        previous_cfg = OmegaConf.create(state["config"])
    for section in ("seed", "world_model", "contract", "data", "loss"):
        if previous_cfg[section] != cfg[section]:
            raise ValueError(f"Resume config mismatch in {section}; keep the original training config")
    for key in (
        "max_train_steps", "warmup_steps", "lr", "start_lr", "final_lr", "weight_decay", "betas",
        "max_grad_norm", "mixed_precision",
    ):
        if previous_cfg.trainer[key] != cfg.trainer[key]:
            raise ValueError(f"Resume config mismatch in trainer.{key}; keep the original training config")
    if path.is_dir():
        accelerator.load_state(str(path))
    else:
        accelerator.unwrap_model(world_model.predictor).load_state_dict(state["predictor"])
        optimizer.load_state_dict(state["optimizer"])
        # Old interval .pt files contain optimizer/step but no scheduler or RNG state.
        scheduler_state = scheduler.state_dict()
        scheduler_state.update(
            last_epoch=state["step"], _step_count=state["step"] + 1,
            _last_lr=[group["lr"] for group in optimizer.param_groups],
        )
        scheduler.load_state_dict(scheduler_state)
        logger.warning(
            "Legacy .pt resume: restored predictor, optimizer and step; reconstructed scheduler. "
            "RNG, sampler and scaler state were not saved, so this resume is not exactly reproducible."
        )
    logger.info("Resumed %s at step %s", path, state["step"])
    return int(state["step"])


def main(cfg):
    accelerator = Accelerator(
        mixed_precision=cfg.trainer.mixed_precision, log_with=cfg.trainer.get("log_with"),
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, data_seed=cfg.seed),
    )
    set_seed(cfg.seed)
    output_dir = Path(cfg.run_root_dir).expanduser() / cfg.run_id
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.disabled = not accelerator.is_main_process
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        for handler in (
            logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ):
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
    try:
        logger.info("Starting JEPA training; log: %s", output_dir / "train.log")
        train(cfg, accelerator, output_dir, logger)
    except Exception:
        logger.exception("Training failed")
        raise
    finally:
        accelerator.end_training()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()


def train(cfg, accelerator, output_dir, logger):
    contract = LatentStepContract(**cfg.contract)
    if cfg.data.get("source") == "lerobot_v3":
        if cfg.data.get("camera_keys") is not None and cfg.data.get("camera_key") is not None:
            raise ValueError("Configure data.camera_keys or data.camera_key, not both")
        num_cameras = len(cfg.data.camera_keys) if cfg.data.get("camera_keys") is not None else 1
        if num_cameras != cfg.world_model.get("num_views", 1):
            raise ValueError("world_model.num_views must match the number of data.camera_keys")
    # The upstream AC predictor uses one width for its action and state encoders.
    widths = (contract.latent_action_dim, contract.state_dim, cfg.world_model.predictor.action_dim)
    if len(set(widths)) != 1:
        raise ValueError(f"latent_action_dim, state_dim and predictor.action_dim must match, got {widths}")
    pretrained_encoder = (
        cfg.world_model.encoder.get("model_path")
        if cfg.world_model.encoder.get("type", "vjepa2") == "qwen3_5"
        else cfg.world_model.get("checkpoint", {}).get("encoder_path")
    )
    if not pretrained_encoder and not cfg.world_model.get("allow_random_encoder", False):
        raise ValueError(
            "A pretrained encoder path is required: set world_model.encoder.model_path for Qwen "
            "or world_model.checkpoint.encoder_path for V-JEPA2. The encoder is frozen, so a random encoder "
            "would train the predictor in a meaningless latent space. Set world_model.allow_random_encoder=true "
            "only for tests."
        )
    world_model = VJEPA2ACWorldModel.from_config(cfg.world_model).freeze_encoder()
    logger.info(
        "Visual input: %s views x %s tokens = %s tokens/frame; camera order: %s",
        world_model.num_views, world_model.tokens_per_view, world_model.tokens_per_frame,
        cfg.data.get("camera_keys", cfg.data.get("camera_key")),
    )
    for report in world_model.load_reports:
        logger.info("%s", report)
    world_model.encoder.to(accelerator.device)

    eval_transition = None
    if cfg.loss.get("eval_state_transition"):
        eval_transition = build_state_transition(cfg.loss.eval_state_transition)

    train_set, val_set = build_datasets(cfg, contract)
    logger.info("%s train / %s val windows", len(train_set), len(val_set))
    if len(train_set) < cfg.data.batch_size * accelerator.num_processes:
        raise ValueError(
            f"Only {len(train_set)} train windows for batch size {cfg.data.batch_size} x {accelerator.num_processes}"
        )
    if len(val_set) == 0:
        raise ValueError("Validation set is empty; increase data.val_fraction or use longer episodes")
    loader_kwargs = dict(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True)
    # Iterator creation must not consume the model's global RNG when resuming mid-epoch.
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(cfg.seed), **loader_kwargs
    )
    val_loader = DataLoader(
        val_set, shuffle=False, drop_last=False, generator=torch.Generator().manual_seed(cfg.seed + 1), **loader_kwargs
    )

    optimizer = torch.optim.AdamW(
        [p for p in world_model.predictor.parameters() if p.requires_grad],
        lr=cfg.trainer.lr,
        betas=tuple(cfg.trainer.betas),
        weight_decay=cfg.trainer.weight_decay,
    )
    # Not passed to accelerator.prepare: an AcceleratedScheduler steps num_processes times per
    # optimizer step, while lr_lambda counts optimizer steps (= trainer.max_train_steps).
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg))
    world_model.predictor, optimizer, train_loader, val_loader = accelerator.prepare(
        world_model.predictor, optimizer, train_loader, val_loader
    )
    accelerator.register_for_checkpointing(scheduler)
    if cfg.trainer.get("log_with"):
        accelerator.init_trackers(cfg.run_id, config=OmegaConf.to_container(cfg, resolve=True))

    step = 0
    if cfg.trainer.get("resume_from_checkpoint"):
        step = load_checkpoint(
            cfg.trainer.resume_from_checkpoint, world_model, accelerator, cfg, optimizer, scheduler, train_loader, logger
        )
    if not 0 <= step <= cfg.trainer.max_train_steps:
        raise ValueError(f"Checkpoint step {step} is outside [0, {cfg.trainer.max_train_steps}]")
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "config.yaml")
    epoch, batches_done = divmod(step, len(train_loader))
    train_loader.set_epoch(epoch)
    epoch_loader = accelerator.skip_first_batches(train_loader, batches_done) if batches_done else train_loader
    data_iter = iter(epoch_loader)
    world_model.predictor.train()
    while step < cfg.trainer.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            train_loader.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        with accelerator.autocast():
            out = compute_losses(world_model, batch, cfg)
        accelerator.backward(out["loss"])
        accelerator.clip_grad_norm_(world_model.predictor.parameters(), cfg.trainer.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % cfg.trainer.log_interval == 0:
            metrics = {f"train/{k}": out[k].item() for k in ("loss", "teacher_forcing", "rollout", "rollout_final")}
            metrics["train/lr"] = scheduler.get_last_lr()[0]
            logger.info("step %s: %s", step, ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            accelerator.log(metrics, step=step)
        if step % cfg.trainer.eval_interval == 0 or step == cfg.trainer.max_train_steps:
            metrics = {f"val/{k}": v for k, v in evaluate(world_model, val_loader, cfg, accelerator).items()}
            if eval_transition is not None:
                deploy = evaluate(world_model, val_loader, cfg, accelerator, eval_transition)
                metrics.update({f"val_deploy_state/{k}": v for k, v in deploy.items()})
            logger.info("step %s: %s", step, ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            accelerator.log(metrics, step=step)
        if step % cfg.trainer.save_interval == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"steps_{step}", accelerator, cfg, step, train_loader
            )
            logger.info("Saved %s", output_dir / "checkpoints" / f"steps_{step}")

    save_checkpoint(output_dir / "checkpoints" / "final", accelerator, cfg, step, train_loader)
    if accelerator.is_main_process:
        # Retain the existing small predictor export for inference / weights-only initialization.
        path = output_dir / "checkpoints" / "final.pt"
        torch.save({
            "predictor": accelerator.unwrap_model(world_model.predictor).state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True), "step": step,
        }, path.with_suffix(".pt.tmp"))
        path.with_suffix(".pt.tmp").replace(path)
    accelerator.wait_for_everyone()
    logger.info("Finished step %s; saved %s", step, output_dir / "checkpoints" / "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="evo_intervention/config/world_model/qwen3_5_piperx.yaml")
    args, overrides = parser.parse_known_args()
    main(OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(overrides)))
