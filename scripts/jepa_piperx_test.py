"""JEPA regression tests: native episode alignment, frozen vision targets and causal prediction."""

import json

import av
import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file
from transformers import Qwen3_5Config
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from evo_intervention.dataloader.world_model_windows import (
    LeRobotV3EpisodeSource, LatentStepContract, TrajectoryWindowDataset, split_episodes,
)
from evo_intervention.model.modules.world_model import VJEPA2ACWorldModel, teacher_forcing_rollout_loss


@pytest.fixture
def recording(tmp_path):
    camera = "observation.images.right_environment_1"
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "features": {camera: {"dtype": "video"}},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    stats = {key: {"min": [0] * 14, "max": [200] * 14} for key in ("observation.state", "action")}
    (tmp_path / "meta/stats.json").write_text(json.dumps(stats))
    rows, episodes = [], []
    for ep, episode_id in enumerate([5, 12]):
        prefix = f"videos/{camera}"
        episodes.append({
            "episode_index": episode_id, "length": 6,
            "data/chunk_index": 0, "data/file_index": 0,
            f"{prefix}/chunk_index": ep, f"{prefix}/file_index": ep * 2,
            f"{prefix}/from_timestamp": 2 / 30,
        })
        for t in range(6):
            rows.append({
                "episode_index": episode_id, "frame_index": t, "timestamp": t / 30,
                "observation.state": np.full(14, 100 + ep * 10 + t, dtype=np.float32),
                "action": np.full(14, 50 + ep * 20 + t, dtype=np.float32),
            })
        video_path = tmp_path / info["video_path"].format(video_key=camera, chunk_index=ep, file_index=ep * 2)
        video_path.parent.mkdir(parents=True)
        with av.open(str(video_path), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for value in [99, 98] + [40 + ep * 40 + t * 4 for t in range(6)]:
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    pd.DataFrame(rows).to_parquet(tmp_path / "data/chunk-000/file-000.parquet")
    pd.DataFrame(episodes).to_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    return LeRobotV3EpisodeSource(tmp_path, camera)


def test_windows_use_camera_offsets_and_absolute_actions_without_crossing_episodes(recording):
    contract = LatentStepContract(actions_per_latent_step=2, action_reduce="last", action_dim=14, state_dim=14)
    dataset = TrajectoryWindowDataset(recording, contract, window_frames=3, episodes=[1])
    assert dataset.index == [(1, 0)]  # a later start would cross the episode boundary
    sample = dataset[0]
    assert sample["frames"].shape == (3, 3, 32, 32)
    np.testing.assert_allclose(sample["frames"].float().mean((1, 2, 3)), [80, 88, 96], atol=2)
    torch.testing.assert_close(sample["states"][:, 0], torch.tensor([0.10, 0.12, 0.14]))
    torch.testing.assert_close(sample["actions"][:, 0], torch.tensor([-0.29, -0.27]))
    assert sample["control_actions"].shape == (2, 2, 14)
    assert sample["valid"].all()


def test_train_validation_split_keeps_episodes_disjoint(recording):
    train, val = split_episodes(len(recording), 0.1, 239)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == [0, 1]


@pytest.fixture
def tiny_qwen(tmp_path):
    config = Qwen3_5Config(vision_config={
        "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
        "out_hidden_size": 48, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2, "num_position_embeddings": 16,
    })
    config.vision_config._attn_implementation = "sdpa"
    visual = Qwen3_5VisionModel(config.vision_config)
    weights = {f"model.visual.{key}": value for key, value in visual.state_dict().items()}
    weights["model.language_model.fake.weight"] = torch.ones(2, 2)
    save_file(weights, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "model.safetensors" for key in weights},
    }))
    config.save_pretrained(tmp_path)
    Qwen2VLImageProcessorFast(
        image_mean=[0.5] * 3, image_std=[0.5] * 3,
        patch_size=16, merge_size=2, temporal_patch_size=2,
    ).save_pretrained(tmp_path)
    cfg = OmegaConf.create({
        "img_size": 64, "patch_size": 32, "tubelet_size": 2, "num_frames": 8,
        "encoder": {"type": "qwen3_5", "model_path": str(tmp_path)},
        "predictor": {"embed_dim": 64, "depth": 1, "num_heads": 4, "action_dim": 14},
    })
    return VJEPA2ACWorldModel.from_config(cfg).freeze_encoder()


def test_only_predictor_receives_gradients_and_qwen_preprocessing_is_correct(tiny_qwen):
    model = tiny_qwen
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not model.encoder.training
    assert not hasattr(model.encoder, "language_model")
    frames = torch.randint(0, 256, (1, 3, 3, 64, 64), dtype=torch.uint8)
    black = model.preprocess(torch.zeros_like(frames))
    torch.testing.assert_close(black, -torch.ones_like(black))
    with torch.no_grad():
        latents = model.encode(model.preprocess(frames))
        singles = torch.cat([model.encode(model.preprocess(frames[:, t:t+1])) for t in range(3)], dim=1)
    assert latents.shape == (1, 3, 4, 48)
    torch.testing.assert_close(latents, singles)
    actions, states = torch.randn(1, 2, 14), torch.randn(1, 3, 14)
    out = teacher_forcing_rollout_loss(model, latents, actions, states, rollout_steps=2)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert all(p.grad is None for p in model.encoder.parameters())
    assert model.predictor.predictor_blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert model.predictor.action_encoder.weight.grad.abs().sum() > 0


def test_predictor_cannot_see_future_frames_states_or_actions(tiny_qwen):
    model = tiny_qwen.eval()
    latents, actions, states = torch.randn(1, 3, 4, 48), torch.randn(1, 3, 14), torch.randn(1, 3, 14)
    expected = model.predict(latents, actions, states)[:, 0]
    latents[:, 1:] += 100
    actions[:, 1:] += 100
    states[:, 1:] += 100
    torch.testing.assert_close(model.predict(latents, actions, states)[:, 0], expected)
