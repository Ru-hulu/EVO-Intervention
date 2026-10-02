"""Three-camera JEPA tests: video offsets, token ordering, causality and real CPU training."""

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from evo_intervention.dataloader.world_model_windows import LeRobotV3EpisodeSource, TrajectoryWindowDataset
from evo_intervention.model.modules.world_model import (
    HoldStateTransition, LatentStepContract, VJEPA2ACWorldModel, teacher_forcing_rollout_loss,
)
from evo_intervention.planning import CEMConfig, GoalLatentScorer, LatentChunkPlanner
from evo_intervention.training import train_world_model
from scripts.jepa_piperx_test import tiny_qwen


CAMERAS = [
    "observation.images.right_environment_1",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]


@pytest.fixture
def recording(tmp_path):
    """Different video offsets and pixel values make camera/time mixups observable."""
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0", "features": {c: {"dtype": "video"} for c in CAMERAS},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    stats = {key: {"min": [0] * 14, "max": [100] * 14} for key in ("observation.state", "action")}
    (tmp_path / "meta/stats.json").write_text(json.dumps(stats))
    meta = {"episode_index": 37, "length": 7, "data/chunk_index": 0, "data/file_index": 0}
    for view, camera in enumerate(CAMERAS):
        offset = view + 1
        prefix = f"videos/{camera}"
        meta.update({f"{prefix}/chunk_index": 0, f"{prefix}/file_index": view,
                     f"{prefix}/from_timestamp": offset / 30})
        path = tmp_path / info["video_path"].format(video_key=camera, chunk_index=0, file_index=view)
        path.parent.mkdir(parents=True)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for value in [220] * offset + [30 + view * 60 + t * 4 for t in range(7)]:
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value, np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    rows = [{"episode_index": 37, "frame_index": t, "timestamp": t / 30,
             "observation.state": np.full(14, 50 + t, np.float32),
             "action": np.full(14, 20 + t, np.float32)} for t in range(7)]
    pd.DataFrame(rows).to_parquet(tmp_path / "data/chunk-000/file-000.parquet")
    pd.DataFrame([meta]).to_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    return tmp_path


def model_config(tiny_qwen):
    return OmegaConf.create({
        "img_size": 64, "patch_size": 32, "tubelet_size": 2, "num_frames": 16, "num_views": 3,
        "encoder": {"type": "qwen3_5", "model_path": tiny_qwen.load_reports[0].path},
        "predictor": {"embed_dim": 64, "depth": 1, "num_heads": 4, "action_dim": 14,
                      "activation_checkpointing": True},
    })


@pytest.fixture
def model(tiny_qwen):
    torch.set_num_threads(2)
    return VJEPA2ACWorldModel.from_config(model_config(tiny_qwen)).freeze_encoder()


def test_three_cameras_follow_order_offsets_and_padding(recording):
    order = [CAMERAS[2], CAMERAS[0], CAMERAS[1]]
    source = LeRobotV3EpisodeSource(recording, camera_keys=order)
    contract = LatentStepContract(actions_per_latent_step=2, action_reduce="last", action_dim=14, state_dim=14)
    dataset = TrajectoryWindowDataset(source, contract, window_frames=5, min_frames=3)
    sample = dataset[0]
    assert sample["frames"].shape == (5, 3, 3, 32, 32)
    expected = np.array([[150 + t * 4, 30 + t * 4, 90 + t * 4] for t in (0, 2, 4, 6)])
    np.testing.assert_allclose(sample["frames"][:4].float().mean((2, 3, 4)), expected, atol=2)
    torch.testing.assert_close(sample["frames"][4], sample["frames"][3])
    assert sample["valid"].tolist() == [True, True, True, True, False]
    torch.testing.assert_close(sample["actions"][:3, 0], torch.tensor([-0.58, -0.54, -0.50]))


@pytest.mark.parametrize("keys", [[], [CAMERAS[0], CAMERAS[0]], CAMERAS[0]])
def test_bad_camera_lists_are_rejected(recording, keys):
    with pytest.raises(ValueError):
        LeRobotV3EpisodeSource(recording, camera_keys=keys)


def test_view_tokens_equal_independent_encodings(model, tiny_qwen):
    frames = torch.randint(0, 256, (2, 3, 3, 3, 48, 48), dtype=torch.uint8)
    with torch.no_grad():
        z = model.encode(model.preprocess(frames))
        expected = torch.cat([tiny_qwen.encode(tiny_qwen.preprocess(frames[:, :, v])) for v in range(3)], dim=2)
    assert z.shape == (2, 3, 12, 48)
    torch.testing.assert_close(z, expected)
    # Changing one camera must leave the encoder targets of the other cameras intact.
    frames[:, :, 1] = 0
    with torch.no_grad():
        changed = model.encode(model.preprocess(frames))
    torch.testing.assert_close(changed[:, :, :4], z[:, :, :4])
    torch.testing.assert_close(changed[:, :, 8:], z[:, :, 8:])
    assert not torch.allclose(changed[:, :, 4:8], z[:, :, 4:8])
    with pytest.raises(ValueError, match="Expected 3 camera views"):
        model.preprocess(frames[:, :, 0])


def test_joint_predictor_is_causal_and_distinguishes_views(model):
    model.eval()
    z, actions, states = torch.randn(2, 3, 12, 48), torch.randn(2, 3, 14), torch.randn(2, 3, 14)
    expected = model.predict(z, actions, states)[:, 0]
    z[:, 1:] += 100
    states[:, 1:] += 100
    actions[:, 1:] += 100
    torch.testing.assert_close(model.predict(z, actions, states)[:, 0], expected)
    attn = model.predictor.predictor_blocks[0].attn
    time, row, col = attn.separate_positions(torch.tensor([0, 4, 8, 12]), H_patches=6, W_patches=2)
    assert time.tolist() == [0, 0, 0, 1]
    assert row.tolist() == [0, 2, 4, 0]
    assert col.tolist() == [0, 0, 0, 0]
    # Three views share one time block with one action token and one state token.
    mask = model.predictor.attn_mask
    assert mask[:14, :14].all() and not mask[:14, 14:28].any()


def test_multiview_loss_rollout_and_planning(model):
    frames = torch.randint(0, 256, (2, 3, 3, 3, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        z = model.encode(model.preprocess(frames))
    actions, states = torch.randn(2, 2, 14), torch.randn(2, 3, 14)
    rollout, _ = model.rollout(z[:, :1], states[:, :1], actions[:, :0], actions, future_states=states[:, 1:2])
    assert rollout.shape == (2, 2, 12, 48)
    torch.testing.assert_close(rollout[:, 0], model.predict(z[:, :-1], actions, states[:, :-1])[:, 0])
    out = teacher_forcing_rollout_loss(model, z, actions, states, rollout_steps=2)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert model.predictor.action_encoder.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    model.eval()
    planner = LatentChunkPlanner(
        model, LatentStepContract(action_dim=14, state_dim=14), HoldStateTransition(), GoalLatentScorer(z[0, -1]),
        CEMConfig(iterations=1, population=4, num_elites=2), horizon_steps=2,
    )
    ctx = planner.observe(frames[:1, :1], states[:1, :1])
    plan = planner.plan(ctx, torch.randn(3, 2, 14), generator=torch.Generator().manual_seed(239))
    assert plan.chunk.shape == (2, 14)
    assert planner.evaluate(ctx, plan.chunk[None]).item() == pytest.approx(plan.cem.best_score, abs=1e-5)


def test_three_camera_training_loop(tmp_path, tiny_qwen, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    torch.set_num_threads(2)
    root = tmp_path / "episodes"
    root.mkdir()
    rng = np.random.default_rng(239)
    for ep in range(4):
        np.savez(root / f"ep_{ep}.npz",
                 frames=rng.integers(0, 256, (9, 3, 64, 64, 3), dtype=np.uint8),
                 states=rng.normal(size=(9, 14)).astype(np.float32),
                 actions=rng.normal(size=(9, 14)).astype(np.float32))
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "evo_intervention/config/world_model/qwen3_5_piperx.yaml")
    cfg.world_model = model_config(tiny_qwen)
    cfg.data = OmegaConf.create({"source": "npz", "root": str(root), "window_frames": 3,
                                "min_frames": None, "val_fraction": 0.25, "batch_size": 2, "num_workers": 0})
    cfg.contract.actions_per_latent_step = 2
    cfg.loss.rollout_steps = 2
    cfg.trainer.update({"max_train_steps": 3, "warmup_steps": 1, "mixed_precision": "no",
                        "log_interval": 1, "eval_interval": 2, "eval_batches": 2, "save_interval": 2})
    cfg.run_root_dir, cfg.run_id = str(tmp_path / "runs"), "three_camera_test"
    train_world_model.main(cfg)
    result = torch.load(Path(cfg.run_root_dir) / cfg.run_id / "checkpoints/final.pt", weights_only=True)
    assert result["step"] == 3 and result["config"]["world_model"]["num_views"] == 3
    assert (Path(cfg.run_root_dir) / cfg.run_id / "checkpoints/steps_2/trainer_state.json").is_file()


def test_configs_require_three_camera_inputs():
    root = Path(__file__).resolve().parents[1]
    names = ["qwen3_5_piperx.yaml", "qwen3_5_piperx_normal.yaml"]
    for name in names:
        cfg = OmegaConf.load(root / "evo_intervention/config/world_model" / name)
        assert cfg.world_model.num_views == len(cfg.data.camera_keys) == 3
        assert "camera_key" not in cfg.data
