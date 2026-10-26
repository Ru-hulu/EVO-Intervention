"""随机小张量检查事件压缩、位置编码和队列梯度；不运行训练循环或优化器。"""

import pytest
import torch

from mem_lite.models.event_encoder import EventEncoder, encode_event_intervals
from mem_lite.models.event_queue import EventQueue


@pytest.fixture
def encoder():
    torch.manual_seed(42)
    return EventEncoder(input_dim=12, hidden_dim=32, num_heads=4).eval()


def test_ragged_events_equal_separate_encoding(encoder):
    events = [
        [torch.randn(3, 12)],
        [torch.randn(4, 12), torch.randn(2, 12), torch.randn(5, 12)],
    ]
    with torch.no_grad():
        together = encoder(events)
        separate = torch.cat([encoder([event]) for event in events])
    assert together.shape == (2, 32, 32)
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
    assert len(encoder.layers) == 4
    assert len({id(layer.attention.in_proj_weight) for layer in encoder.layers}) == 4


def test_frame_order_matters_and_event_batch_order_does_not(encoder):
    frames = [torch.randn(3, 12), torch.randn(3, 12)]
    with torch.no_grad():
        forward = encoder([frames])[0]
        reverse = encoder([list(reversed(frames))])[0]
        swapped_batch = encoder([list(reversed(frames)), frames])
    assert (forward - reverse).abs().max() > 1e-4
    torch.testing.assert_close(swapped_batch[1], forward, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(swapped_batch[0], reverse, atol=1e-6, rtol=1e-5)


def test_intervals_use_queue_indices_and_exclude_outside_frames(encoder):
    frames = [torch.randn(3, 12) for _ in range(5)]
    with torch.no_grad():
        actual = encode_event_intervals(encoder, frames, [(1, 3), (3, 5)])
        expected = encoder([frames[1:3], frames[3:5]])
        frames[0] = torch.randn(7, 12) * 100
        unrelated_change = encode_event_intervals(encoder, frames, [(1, 3), (3, 5)])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, unrelated_change)
    for bad in [[(2, 2)], [(-1, 3)], [(1, 6)]]:
        with pytest.raises(ValueError, match="观测队列"):
            encode_event_intervals(encoder, frames, bad)


def test_queue_positions_prefixes_and_video_isolation():
    first, second = torch.randn(32, 16), torch.randn(32, 16)
    original_first, original_second = first.clone(), second.clone()
    queue, another_video = EventQueue(), EventQueue()
    queue.append(first)
    prefix = queue.as_tokens()
    queue.append(second)
    all_tokens = queue.as_tokens()
    assert all_tokens.shape == (64, 16)
    torch.testing.assert_close(queue.as_tokens(end=1), prefix)
    torch.testing.assert_close(all_tokens[:32], prefix)
    # 同一事件所有 tokens 的位置偏移相同，相邻事件的位置偏移不同。
    offset0, offset1 = prefix - first, all_tokens[32:] - second
    torch.testing.assert_close(offset0, offset0[:1].expand_as(offset0))
    torch.testing.assert_close(offset1, offset1[:1].expand_as(offset1))
    assert not torch.allclose(offset0, offset1)
    torch.testing.assert_close(queue.as_tokens(), all_tokens)
    torch.testing.assert_close(first, original_first)
    torch.testing.assert_close(second, original_second)
    assert len(another_video) == 0
    queue.clear()
    assert len(queue) == 0
    queue.append(first)
    torch.testing.assert_close(queue.as_tokens(), prefix)


def test_queue_keeps_gradient_to_all_compression_layers(encoder):
    # 仅做一次梯度连通性检查，不执行优化器更新。
    frames = [torch.randn(2, 12), torch.randn(3, 12), torch.randn(4, 12)]
    events = encode_event_intervals(encoder, frames, [(0, 1), (1, 3)])
    events.retain_grad()
    queue = EventQueue()
    queue.append(events[0])
    prefix1 = queue.as_tokens()
    queue.append(events[1])
    prefix2 = queue.as_tokens()
    loss = (prefix1 * torch.randn_like(prefix1)).mean() + (prefix2 * torch.randn_like(prefix2)).mean()
    loss.backward()
    assert all(event_grad.abs().sum() > 0 for event_grad in events.grad)
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert all(frame.grad is None for frame in frames)


def test_bfloat16_features_work_with_fp32_encoder(encoder):
    with torch.no_grad():
        result = encoder([[torch.randn(3, 12, dtype=torch.bfloat16)]])
    assert result.shape == (1, 32, 32)
    assert result.dtype == torch.float32
    assert result.isfinite().all()


def test_empty_events_and_uncompressed_queue_entries_are_rejected(encoder):
    for events in ([], [[]], [[torch.empty(0, 12)]]):
        with pytest.raises(ValueError, match="事件"):
            encoder(events)
    queue = EventQueue()
    with pytest.raises(ValueError, match="非空"):
        queue.as_tokens()
    with pytest.raises(ValueError, match="压缩表征"):
        queue.append(torch.randn(100, 32))
