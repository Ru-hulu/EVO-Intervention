"""M2 固定历史长度、事件顺序、变长批次与跨模块梯度检查，不运行训练。"""

import pytest
import torch

from mem_lite.models.event_encoder import EventEncoder
from mem_lite.models.event_queue import EventQueue
from mem_lite.models.m2_compressor import M2Compressor


@pytest.fixture
def compressor():
    torch.manual_seed(12)
    return M2Compressor(input_dim=16, hidden_dim=32, num_heads=4).eval()


def test_fixed_128_tokens_and_ragged_batch_matches_single(compressor):
    # 用双精度检查遮罩语义，避免不同矩阵批尺寸的 FP32 累积舍入误差。
    compressor.double()
    queues = [torch.randn(count * 32, 16, dtype=torch.float64) for count in (1, 5, 12)]
    with torch.no_grad():
        batched = compressor(queues)
        singles = torch.cat([compressor([queue]) for queue in queues])
    assert batched.shape == (3, 128, 32)
    torch.testing.assert_close(batched, singles, atol=1e-10, rtol=1e-9)
    assert len(compressor.layers) == 4
    assert len({id(layer.attention.in_proj_weight) for layer in compressor.layers}) == 4


def test_event_order_changes_history_but_later_calls_do_not_change_prefix(compressor):
    first, second = torch.randn(32, 16), torch.randn(32, 16)
    queue = EventQueue()
    queue.append(first)
    with torch.no_grad():
        prefix = compressor([queue.as_tokens()])
        queue.append(second)
        full = compressor([queue.as_tokens()])
        reread = compressor([queue.as_tokens(end=1)])
        reverse_queue = EventQueue()
        reverse_queue.append(second)
        reverse_queue.append(first)
        reversed_history = compressor([reverse_queue.as_tokens()])
    torch.testing.assert_close(prefix, reread)
    assert (full - prefix).abs().max() > 1e-4
    assert (full - reversed_history).abs().max() > 1e-4


def test_changing_earliest_event_changes_history(compressor):
    # 即使队列较长，最早的事件也必须仍是 M2 的输入，不能只取最新事件。
    queue = torch.randn(10 * 32, 16)
    changed = queue.clone()
    changed[:32] = torch.randn(32, 16) * 3
    with torch.no_grad():
        histories = compressor([queue, changed])
    assert (histories[0] - histories[1]).abs().max() > 1e-4


def test_history_gradient_reaches_encoder_and_every_m2_layer(compressor):
    encoder = EventEncoder(input_dim=8, hidden_dim=16, num_heads=2).eval()
    encoded = encoder([[torch.randn(3, 8)], [torch.randn(2, 8), torch.randn(4, 8)]])
    encoded.retain_grad()
    queue = EventQueue()
    queue.append(encoded[0])
    prefix = queue.as_tokens()
    queue.append(encoded[1])
    history = compressor([prefix, queue.as_tokens()])
    # 仅检测梯度连接，不构造文本标签、不执行优化器更新。
    loss = (history * torch.randn_like(history)).mean()
    loss.backward()
    assert encoded.grad.isfinite().all()
    assert all(gradient.abs().sum() > 0 for gradient in encoded.grad)
    for model in (encoder, compressor):
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and parameter.grad.isfinite().all(), name
            assert parameter.grad.abs().sum() > 0, name
    encoder_parameters = {id(parameter) for parameter in encoder.parameters()}
    assert not encoder_parameters.intersection(id(parameter) for parameter in compressor.parameters())


def test_bfloat16_queue_and_empty_input(compressor):
    with torch.no_grad():
        history = compressor([torch.randn(32, 16, dtype=torch.bfloat16)])
    assert history.shape == (1, 128, 32) and history.isfinite().all()
    for queues in ([], [torch.empty(0, 16)]):
        with pytest.raises(ValueError, match="非空事件队列"):
            compressor(queues)
