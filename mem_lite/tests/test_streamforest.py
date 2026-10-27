"""真实合并、预算约束、事件边界及 Qwen 视觉接入的离线检查。"""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from mem_lite.memory import MemoryManager
from mem_lite.memory.tome import merge_tokens
from mem_lite.memory.vision import QwenFrameEncoder


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weighted_merge_preserves_mean_and_handles_zero_features(dtype):
    torch.manual_seed(7)
    tokens = torch.randn(17, 16).to(dtype)
    merged = merge_tokens(tokens, 1)
    torch.testing.assert_close(merged.float()[0], tokens.float().mean(0), atol=0.003, rtol=0.02)
    assert merged.dtype == dtype
    assert merge_tokens(tokens, 32) is tokens
    assert merge_tokens(torch.zeros_like(tokens), 3).eq(0).all()


def test_long_video_stays_within_budget_with_variable_grids():
    config = json.loads((Path(__file__).parents[1] / "configs/streamforest.json").read_text())
    memory = MemoryManager(16, **config)
    assert memory.token_budget == 1008
    assert memory.get_memory_tokens() is None
    torch.manual_seed(12)
    for index in range(240):
        frame = torch.randn((4, 20, 81)[index % 3], 16, requires_grad=True)
        memory.update(frame)
        history = memory.get_memory_tokens()
        assert 0 < len(history) <= memory.token_budget
        assert sum(len(event.tokens) for event in memory.memory_long) <= 800
        assert not history.requires_grad and torch.isfinite(history).all()
    assert any(event.merges > 1 for event in memory.memory_long)
    torch.testing.assert_close(memory.get_memory_tokens(), history)
    memory.reset()
    assert memory.get_memory_tokens() is None
    memory.update(frame)
    torch.testing.assert_close(memory.get_memory_tokens(), merge_tokens(frame, 64))


def test_event_boundary_follows_visual_change_and_tiny_quota_is_enforced():
    memory = MemoryManager(8, st_memory_windows=(1, 2), st_memory_tokens=(8, 4),
                           event_split_window=3, long_memory_tokens_per_frame=2, long_memory_tokens_quota=1)
    # 最早窗口的相似度为 [1, -1, 1]，应在第2帧后切分，而非固定窗口边界。
    for sign in (1, 1, -1, -1, -1, -1):
        memory.update(torch.full((8, 8), float(sign)))
    assert memory.frames_in_long == 2
    assert len(memory.memory_long) == 1
    assert len(memory.memory_long[0].tokens) == 1
    assert len(memory.get_memory_tokens()) <= memory.token_budget


def test_qwen_rectangular_grids_freeze_and_batch_independence(checkpoint):
    visual = checkpoint[1].model.visual
    encoder = QwenFrameEncoder(visual)
    encoder.train()
    assert encoder.visual is visual
    assert not any(module.training for module in encoder.modules())
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
    config = visual.config
    processor = Qwen2VLImageProcessor(
        min_pixels=1024, max_pixels=8192, patch_size=config.patch_size,
        temporal_patch_size=config.temporal_patch_size, merge_size=config.spatial_merge_size,
    )
    images = [Image.new("RGB", size, color) for size, color in
              (((32, 64), "red"), ((64, 32), "blue"), ((64, 64), "green"))]
    inputs = processor(images=images, return_tensors="pt")
    positions = [510, 511, 512]
    together = encoder(inputs["pixel_values"], inputs["image_grid_thw"], positions)
    counts = (inputs["image_grid_thw"].prod(-1) // config.spatial_merge_size**2).tolist()
    assert [len(frame) for frame in together] == counts
    memory = MemoryManager(encoder.hidden_size)
    for index, image in enumerate(images):
        single = processor(images=[image], return_tensors="pt")
        frame = encoder(single["pixel_values"], single["image_grid_thw"], [positions[index]])[0]
        torch.testing.assert_close(frame, together[index], atol=1e-6, rtol=1e-5)
        assert frame.shape[1] == encoder.hidden_size and not frame.requires_grad
        memory.update(frame)
    assert memory.get_memory_tokens().shape[1] == config.out_hidden_size
    later = encoder(inputs["pixel_values"], inputs["image_grid_thw"], [value + 512 for value in positions])
    assert not torch.allclose(together[0], later[0])
    temporal_dim = encoder.hidden_size // 4
    torch.testing.assert_close(together[0][:, temporal_dim:], later[0][:, temporal_dim:])
