"""Train the V-JEPA 2-AC predictor on robot trajectory windows with L_teacher-forcing + L_rollout.

The encoder stays frozen; only the predictor is optimized (random init, or the official AC
predictor when `world_model.checkpoint.predictor_path` is set). Plain Accelerate/DDP, no DeepSpeed.

    accelerate launch evo_intervention/training/train_world_model.py \
        --config_yaml evo_intervention/config/world_model/vjepa2_ac_vitg.yaml loss.rollout_steps=4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from evo_intervention.dataloader.world_model_windows import NpzEpisodeSource, TrajectoryWindowDataset, split_episodes
from evo_intervention.model.modules.world_model import (
    LatentStepContract,
    VJEPA2ACWorldModel,
    build_state_transition,
    teacher_forcing_rollout_loss,
)


def build_source(cfg):
    if cfg.data.source == "npz":
        return NpzEpisodeSource(cfg.data.root)
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


def save_checkpoint(path: Path, world_model, accelerator, cfg, step, optimizer=None):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "predictor": accelerator.unwrap_model(world_model.predictor).state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "step": step,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)
    accelerator.print(f"Saved {path}")


def main(cfg):
    accelerator = Accelerator(mixed_precision=cfg.trainer.mixed_precision, log_with=cfg.trainer.get("log_with"))
    set_seed(cfg.seed)
    output_dir = Path(cfg.run_root_dir) / cfg.run_id
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")

    contract = LatentStepContract(**cfg.contract)
    # The upstream AC predictor uses one width for its action and state encoders.
    widths = (contract.latent_action_dim, contract.state_dim, cfg.world_model.predictor.action_dim)
    if len(set(widths)) != 1:
        raise ValueError(f"latent_action_dim, state_dim and predictor.action_dim must match, got {widths}")
    if not cfg.world_model.get("checkpoint", {}).get("encoder_path") and not cfg.world_model.get(
        "allow_random_encoder", False
    ):
        raise ValueError(
            "world_model.checkpoint.encoder_path is required: the encoder is frozen, so a random encoder "
            "would train the predictor in a meaningless latent space. Set world_model.allow_random_encoder=true "
            "only for tests."
        )
    world_model = VJEPA2ACWorldModel.from_config(cfg.world_model).freeze_encoder()
    for report in world_model.load_reports:
        accelerator.print(report)
    world_model.encoder.to(accelerator.device)

    eval_transition = None
    if cfg.loss.get("eval_state_transition"):
        eval_transition = build_state_transition(cfg.loss.eval_state_transition)

    train_set, val_set = build_datasets(cfg, contract)
    accelerator.print(f"{len(train_set)} train / {len(val_set)} val windows")
    if len(train_set) < cfg.data.batch_size * accelerator.num_processes:
        raise ValueError(
            f"Only {len(train_set)} train windows for batch size {cfg.data.batch_size} x {accelerator.num_processes}"
        )
    if len(val_set) == 0:
        raise ValueError("Validation set is empty; increase data.val_fraction or use longer episodes")
    loader_kwargs = dict(batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)

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
    if cfg.trainer.get("log_with"):
        accelerator.init_trackers(cfg.run_id, config=OmegaConf.to_container(cfg, resolve=True))

    step, data_iter = 0, iter(train_loader)
    world_model.predictor.train()
    while step < cfg.trainer.max_train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
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
            accelerator.print(f"step {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            accelerator.log(metrics, step=step)
        if step % cfg.trainer.eval_interval == 0 or step == cfg.trainer.max_train_steps:
            metrics = {f"val/{k}": v for k, v in evaluate(world_model, val_loader, cfg, accelerator).items()}
            if eval_transition is not None:
                deploy = evaluate(world_model, val_loader, cfg, accelerator, eval_transition)
                metrics.update({f"val_deploy_state/{k}": v for k, v in deploy.items()})
            accelerator.print(f"step {step}: " + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))
            accelerator.log(metrics, step=step)
        if step % cfg.trainer.save_interval == 0:
            save_checkpoint(
                output_dir / "checkpoints" / f"steps_{step}.pt", world_model, accelerator, cfg, step, optimizer
            )

    save_checkpoint(output_dir / "checkpoints" / "final.pt", world_model, accelerator, cfg, step)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", default="evo_intervention/config/world_model/vjepa2_ac_vitg.yaml")
    args, overrides = parser.parse_known_args()
    main(OmegaConf.merge(OmegaConf.load(args.config_yaml), OmegaConf.from_dotlist(overrides)))
