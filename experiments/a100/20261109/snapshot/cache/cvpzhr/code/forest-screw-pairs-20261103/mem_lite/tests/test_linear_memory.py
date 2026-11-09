"""真实微型 Qwen 的状态续接、padding、因果性、跨帧梯度和采样边界。"""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from mem_lite.datasets.sequence import EpisodeWindowSampler
from mem_lite.models.linear_memory import gated_delta_forward
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.tests.test_m3_predictor import collator as collator, model as model, sample


def assert_state_equal(a, b, **kwargs):
    assert a.layers.keys() == b.layers.keys()
    for index in a.layers:
        for x, y in zip(a.layers[index], b.layers[index], strict=True):
            torch.testing.assert_close(x, y, **kwargs)


def test_multi_token_continuation_matches_native_contiguous_linear_attention(model):
    layer = model.qwen.model.language_model.layers[0].decoder.linear_attn
    # 较慢遗忘，保证测试确实依赖边界之前的状态。
    with torch.no_grad():
        layer.A_log.fill_(-4)
    hidden = torch.randn(2, 29, 32)
    valid = torch.ones(2, 29, dtype=torch.bool)
    with torch.no_grad():
        expected = layer(hidden)
        whole, conv, recurrent = gated_delta_forward(layer, hidden, valid)
        parts, old_conv, old_recurrent = [], None, None
        for start, end in [(0, 1), (1, 8), (8, 19), (19, 29)]:
            out, old_conv, old_recurrent = gated_delta_forward(
                layer, hidden[:, start:end], valid[:, start:end], old_conv, old_recurrent)
            parts.append(out)
        reset, _, _ = gated_delta_forward(layer, hidden[:, 19:], valid[:, 19:])
    torch.testing.assert_close(whole, expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(torch.cat(parts, dim=1), expected, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(old_conv, conv)
    torch.testing.assert_close(old_recurrent, recurrent, atol=3e-6, rtol=3e-5)
    assert not torch.allclose(parts[-1], reset, atol=1e-5, rtol=1e-4)


def test_padding_and_inactive_streams_do_not_change_linear_states(model):
    layer = model.qwen.model.language_model.layers[0].decoder.linear_attn
    hidden = torch.randn(2, 11, 32)
    valid = torch.tensor([[True]*11, [False, True, True, False, True, False, False, True, False, False, False]])
    hidden[~valid] = float("nan")
    with torch.no_grad():
        _, conv, recurrent = gated_delta_forward(layer, torch.randn(2, 5, 32), torch.ones(2, 5, dtype=torch.bool))
        mixed, next_conv, next_recurrent = gated_delta_forward(layer, hidden, valid, conv, recurrent)
        for i in range(2):
            selected = hidden[i:i+1, valid[i]]
            single, c, s = gated_delta_forward(layer, selected, torch.ones(1, selected.shape[1], dtype=torch.bool),
                                              conv[i:i+1], recurrent[i:i+1])
            torch.testing.assert_close(mixed[i, valid[i]], single[0], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(next_conv[i:i+1], c)
            torch.testing.assert_close(next_recurrent[i:i+1], s, atol=3e-6, rtol=3e-5)
        _, c, s = gated_delta_forward(layer, hidden, torch.zeros_like(valid), conv, recurrent)
    torch.testing.assert_close(c, conv, atol=0, rtol=0)
    torch.testing.assert_close(s, recurrent, atol=0, rtol=0)


def test_window_and_single_frame_predictions_match_with_no_full_kv(model, collator):
    batches = [collator([sample(torch.randn(7, 16)), sample(prompt="另一条独立视频")]) for _ in range(3)]
    attn = model.qwen.model.language_model.layers[1].self_attn
    with patch.object(attn, "forward", wraps=attn.forward) as full, torch.no_grad():
        window = model(batches)
        state, outputs = None, []
        for batch in batches:
            result = model(batch, state=state)
            outputs.append(result["bin_logits"])
            state = result["state"]
    torch.testing.assert_close(window["bin_logits"], torch.cat(outputs), atol=0, rtol=0)
    assert_state_equal(window["state"], state, atol=0, rtol=0)
    assert all(call.kwargs.get("past_key_values") is None for call in full.call_args_list)
    assert set(state.layers) == {0}  # 没有 Full Attention 层的缓存。
    for row in range(2):
        single_state = None
        with torch.no_grad():
            for t in range(3):
                item = sample(batches[t]["history_tokens"][row, :7]) if row == 0 else sample(prompt="另一条独立视频")
                result = model(collator([item]), state=single_state)
                single_state = result["state"]
                torch.testing.assert_close(result["bin_logits"][0], window["bin_logits"][t * 2 + row],
                                           atol=3e-6, rtol=3e-5)


def test_future_frames_and_future_history_do_not_affect_previous_outputs(model, collator):
    batches = [collator([sample(torch.randn(5, 16))]) for _ in range(4)]
    changed = [*batches[:2], collator([sample(torch.randn(30, 16), prompt="不同的未来帧")]), batches[3]]
    with torch.no_grad():
        original, altered = model(batches), model(changed)
    torch.testing.assert_close(original["bin_logits"][:2], altered["bin_logits"][:2], atol=0, rtol=0)


def test_state_is_fixed_size_and_explicitly_resettable(model, collator):
    batch = collator([sample()])
    with torch.no_grad():
        first = model(batch)
        state = first["state"]
        for _ in range(12):
            state = model(batch, state=state)["state"]
        fresh = model(batch)
    assert_state_equal(first["state"], fresh["state"], atol=0, rtol=0)
    for i in state.layers:
        assert [x.shape for x in state.layers[i]] == [x.shape for x in first["state"].layers[i]]
    assert all(t.isfinite().all() for pair in state.layers.values() for t in pair)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_gradient_through_states_and_detach_boundary(model, collator, checkpointing):
    model.train()
    model.qwen.model.language_model.gradient_checkpointing = checkpointing
    with torch.no_grad():
        model.qwen.model.language_model.layers[0].decoder.linear_attn.A_log.fill_(-6)
    first = model(collator([sample(torch.randn(4, 16))]))
    previous = first["state"].layers[0][1]
    previous.retain_grad()
    second = model(collator([sample(torch.randn(9, 16))]), torch.tensor([0.7]), state=first["state"])
    second["loss"].backward()
    assert previous.grad is not None and previous.grad.abs().sum() > 0
    detached = second["state"].detach()
    assert all(t.grad_fn is None and not t.requires_grad for pair in detached.layers.values() for t in pair)
    model.zero_grad(set_to_none=True)
    model(collator([sample()]), torch.tensor([0.4]), state=detached)["loss"].backward()
    assert model.progress_head.weight.grad.isfinite().all()


def test_stateful_checkpointing_matches_eager_gradients(model, collator):
    eager = copy.deepcopy(model).train()
    model.train()
    model.qwen.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    batches = [collator([sample(torch.randn(4, 16))]) for _ in range(3)]
    for m in (model, eager):
        m(batches, torch.tensor([0.1, 0.4, 0.8]))["loss"].backward()
    for p, q in zip(model.parameters(), eager.parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5)


def test_distributed_sequence_sampler_has_no_duplicates_and_aligned_boundaries():
    lengths = [1, 3, 9, 2, 6, 4, 11]
    dataset = SimpleNamespace(samples=[(episode, {"frame_index": frame}, None)
                                      for episode, length in enumerate(lengths) for frame in range(length)])
    samplers = [EpisodeWindowSampler(dataset, 2, 3, rank, 3, 41) for rank in range(3)]
    for epoch in [0, 1]:
        for s in samplers:
            s.set_epoch(epoch)
        plans = [list(s) for s in samplers]
        assert len({len(p) for p in plans}) == 1
        seen = []
        for plan, sampler in zip(plans, samplers, strict=True):
            assert len(plan) == len(sampler)
            slots = {}
            for w in plan:
                if w["reset"]:
                    slots = {}
                for row in w["indices"]:
                    for slot, index in enumerate(row):
                        if index is None:
                            continue
                        episode, sample_row, _ = dataset.samples[index]
                        if slot in slots:
                            old_episode, old_frame = slots[slot]
                            assert episode == old_episode and sample_row["frame_index"] == old_frame + 1
                        else:
                            assert sample_row["frame_index"] == 0
                        slots[slot] = (episode, sample_row["frame_index"])
                        seen.append(index)
            # 每个允许存档的组结束后，恢复的下一窗口必定 reset。
            for i, w in enumerate(plan[:-1]):
                if w["group_end"]:
                    assert plan[i+1]["reset"]
        assert sorted(seen) == list(range(len(dataset.samples)))
        assert all([(w["reset"], w["group_end"]) for w in p] ==
                   [(w["reset"], w["group_end"]) for w in plans[0]] for p in plans)


def ddp_sequence_worker(rank, rendezvous, model_path, inputs_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(19)
        model = M3ProgressPredictor(model_path, history_dim=16, dtype=torch.float32, lora_rank=2, lora_alpha=4,
                                    lora_dropout=0, gradient_checkpointing=False, cross_attention_layers=(1,),
                                    cross_attention_dim=16, cross_attention_heads=2).train()
        with torch.no_grad():
            model.memory_adapters["1"].gate.fill_(0.2)
        reference = copy.deepcopy(model)
        wrapped = DistributedDataParallel(model, broadcast_buffers=False, find_unused_parameters=True)
        plans = torch.load(inputs_path, weights_only=True)
        state = None
        for t, batches in enumerate(plans[rank]):
            with wrapped.no_sync() if t == 0 else nullcontext():
                out = wrapped(batches, torch.full((len(batches),), 0.4), state=state)
                out["loss_sum"].backward()
            state = out["state"].detach()
        for plan in plans:
            state = None
            for batches in plan:
                out = reference(batches, torch.full((len(batches),), 0.4), state=state)
                (out["loss_sum"] / 2).backward()
                state = out["state"].detach()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, atol=3e-6, rtol=3e-5)
    finally:
        dist.destroy_process_group()


def test_ddp_stateful_windows_and_inactive_rank(checkpoint, collator, tmp_path):
    plans = []
    for rank in range(2):
        windows = []
        for t in range(2):
            batches = []
            for _ in range(2):
                history = torch.randn(4, 16) if rank == 0 and t == 0 else None
                batch = collator([sample(history)])
                batch["sample_valid"] = torch.tensor([not (rank == 1 and t == 1)])
                batches.append(batch)
            windows.append(batches)
        plans.append(windows)
    path = tmp_path / "inputs.pt"
    torch.save(plans, path)
    mp.spawn(ddp_sequence_worker, args=(str(tmp_path / "gloo"), str(checkpoint[0]), str(path)), nprocs=2, join=True)
