"""Tiny real-Qwen tests for temporal causality, memory writing and TBPTT."""

import copy
import json

import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor, Qwen3_5ForConditionalGeneration

from mem_lite.datasets.vst_dataset import EpisodeWindows, VSTCollator, native_positions
from mem_lite.models.vst_predictor import VSTProgressPredictor
from mem_lite.train_vst import load_trainable_state, trainable_state


@pytest.fixture
def collator(checkpoint):
    config = checkpoint[1].config.vision_config
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=1024, max_pixels=4096, patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size, merge_size=config.spatial_merge_size),
        video_processor=Qwen3VLVideoProcessor(), tokenizer=AutoTokenizer.from_pretrained(checkpoint[0]),
    )
    processor.chat_template = processor.tokenizer.chat_template
    return VSTCollator(processor, compression_ratio=2)


@pytest.fixture(scope="module")
def vst_checkpoint(checkpoint, tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-vst-multilayer")
    config = copy.deepcopy(checkpoint[1].config)
    config.text_config.num_hidden_layers = 4
    config.text_config.layer_types = ["linear_attention", "full_attention"] * 2
    torch.manual_seed(42)
    Qwen3_5ForConditionalGeneration(config).save_pretrained(path)
    return path


@pytest.fixture
def model(vst_checkpoint):
    return VSTProgressPredictor(vst_checkpoint, dtype=torch.float32, lora_rank=2, lora_alpha=4,
                                lora_dropout=0, gradient_checkpointing=False).eval()


def sample(color="red", prompt="放下物体", valid=True):
    return {"images": [Image.new("RGB", (64, 64), color), Image.new("RGB", (32, 64), "blue"),
                       Image.new("RGB", (64, 32), "green")], "task_prompt": prompt, "frame_valid": valid}


def test_positions_and_vst_count(collator, checkpoint):
    items = [sample(), sample(prompt="先拿起然后放下物体")]
    original = collator.base(items)
    native, _ = checkpoint[1].model.get_rope_index(original["input_ids"], original["mm_token_type_ids"],
        image_grid_thw=original["image_grid_thw"], attention_mask=original["attention_mask"])
    for row in range(2):
        valid = original["attention_mask"][row].bool()
        calculated = native_positions(original["mm_token_type_ids"][row, valid],
                                       original["image_grid_thw"][row * 3:row * 3 + 3], collator.merge)
        torch.testing.assert_close(calculated, native[:, row, valid])
    batch = collator(items)
    assert batch["vst_valid"].sum(-1).tolist() == [2, 2]
    assert batch["vst_mask"].sum() == 4
    for row in range(2):
        image_positions = (batch["input_ids"][row] == checkpoint[1].config.image_token_id).nonzero().flatten()
        assert batch["vst_indices"][row, -1] < image_positions[4]  # before both wrists


def test_vst_qkv_are_independent_and_native_o_is_shared(model, collator):
    batch = collator([sample(), sample("green")], 1, 2)
    attention = model.qwen.model.language_model.layers[model.full_layers[0]].self_attn
    assert set(attention.vst_projections) == {"q_proj", "k_proj", "v_proj"}
    for name, projection in attention.vst_projections.items():
        original = getattr(attention.native, name).weight
        torch.testing.assert_close(projection.weight, original, atol=0, rtol=0)
        assert projection.weight.data_ptr() != original.data_ptr()
        assert projection.weight.requires_grad and not original.requires_grad
    calls = []
    def record_output(module, args, output):
        calls.append(output)
    handle = attention.native.o_proj.register_forward_hook(record_output)
    try:
        model(batch)
    finally:
        handle.remove()
    # One shared O call covers every position, including all VST positions.
    assert len(calls) == 1
    assert calls[0].shape[:2] == batch["input_ids"].shape
    calls[0][batch["vst_mask"]].square().sum().backward()
    shared_lora = attention.native.o_proj.lora_B["default"].weight
    assert shared_lora.grad is not None and shared_lora.grad.abs().sum() > 0
    assert attention.native.o_proj.weight.grad is None
    assert not any("vst_projections.o_proj" in name for name in trainable_state(model))


def test_parallel_window_matches_sequential_with_padding_and_cached_prefix(model, collator):
    streams = [[sample("red"), sample("blue"), sample("green")],
               [sample("green", "较长的任务指令"), sample("red", "较长的任务指令"), sample("blue", "较长的任务指令")]]
    with torch.no_grad():
        prefix = model(collator([sample(), sample(prompt="较长的任务指令")]))["cache"]
        parallel = model(collator(sum(streams, []), stream_batch=2, window_size=3), cache=prefix)
        sequential, cache = [], prefix
        for t in range(3):
            result = model(collator([stream[t] for stream in streams]), cache=cache)
            sequential.append(result["bin_logits"])
            cache = result["cache"]
    expected = torch.stack(sequential, dim=1).flatten(0, 1)
    torch.testing.assert_close(parallel["bin_logits"], expected, atol=3e-6, rtol=3e-5)
    assert parallel["cache"].lengths.tolist() == [8, 8]
    for i, (k, v) in cache.layers.items():
        torch.testing.assert_close(parallel["cache"].layers[i][0], k, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(parallel["cache"].layers[i][1], v, atol=3e-6, rtol=3e-5)


def test_future_and_wrist_information_cannot_change_past_memory(model, collator):
    frames = [sample("red"), sample("green"), sample("blue")]
    changed = copy.deepcopy(frames)
    changed[2] = sample("white")
    wrists = copy.deepcopy(frames)
    for frame in wrists:
        frame["images"][1:] = [Image.new("RGB", (32, 64), "black"), Image.new("RGB", (64, 32), "white")]
    with torch.no_grad():
        first = model(collator(frames, 1, 3))
        future = model(collator(changed, 1, 3))
        wrist = model(collator(wrists, 1, 3))
        fresh = model(collator([frames[1]]))
        remembered = model(collator([frames[1]]), cache=model(collator([frames[0]]))["cache"])
    torch.testing.assert_close(first["bin_logits"][:2], future["bin_logits"][:2], atol=0, rtol=0)
    for i, (k, v) in first["cache"].layers.items():
        torch.testing.assert_close(k, wrist["cache"].layers[i][0], atol=0, rtol=0)
        torch.testing.assert_close(v, wrist["cache"].layers[i][1], atol=0, rtol=0)
    assert not torch.allclose(fresh["bin_logits"], remembered["bin_logits"], atol=1e-7, rtol=1e-6)


def test_gradient_reaches_previous_frame_and_detach_stops_it(model, collator):
    model.train()
    first = model(collator([sample()]))
    k, v = next(iter(first["cache"].layers.values()))
    k.retain_grad(); v.retain_grad()
    next_result = model(collator([sample("green")]), torch.tensor([0.7]), first["cache"])
    next_result["loss"].backward()
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert v.grad is not None and v.grad.abs().sum() > 0
    assert model.vst_embedding.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    first = model(collator([sample()]))
    k, v = next(iter(first["cache"].layers.values()))
    k.retain_grad(); v.retain_grad()
    model(collator([sample("green")]), torch.tensor([0.7]), first["cache"].detach())["loss"].backward()
    assert k.grad is None and v.grad is None


@pytest.mark.parametrize("checkpointing", [True, "full_attention"])
def test_checkpointed_window_gradients_match_and_frozen_weights_stay_frozen(model, collator, checkpointing):
    model.train()
    batch = collator([sample(), sample("green"), sample("blue", valid=False)], 1, 3)
    target = torch.tensor([0.1, 0.6, 0.0])
    plain = model(batch, target)
    plain["loss"].backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert plain["cache"].lengths.tolist() == [4]
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing = checkpointing
    checked = model(batch, target)
    checked["loss"].backward()
    torch.testing.assert_close(checked["loss"], plain["loss"], atol=0, rtol=0)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected[name], atol=2e-6, rtol=2e-5)
        else:
            assert parameter.grad is None, name
    assert not model.qwen.model.visual.training


def test_epoch_schedule_exact_coverage_no_cross_episode_and_equal_ddp_steps(tmp_path):
    paths = []
    for episode, length in enumerate([1, 2, 3, 6, 7, 9, 13]):
        path = tmp_path / str(episode); path.mkdir(); paths.append(str(path))
        (path / "summary.json").write_text(json.dumps({"format_version": "streamforest-m3-data-v1",
            "history_dim": 16, "history_capacity": 0, "history_source": {}, "task_prompt": "test"}))
        (path / "samples.jsonl").write_text("\n".join(json.dumps({"frame_index": i, "pt": 0}) for i in range(length)))
    schedules = [EpisodeWindows(paths, 2, 4, rank=r, world_size=3) for r in range(3)]
    assert len(set(map(len, schedules))) == 1
    indices = [i for s in schedules for w in s.windows for i in w["indices"] if i is not None]
    assert sorted(indices) == list(range(41))
    for schedule in schedules:
        previous = {}
        for window in schedule.windows:
            if window["reset"]:
                previous = {}
            for stream in range(2):
                for i in window["indices"][stream * 4:(stream + 1) * 4]:
                    if i is None:
                        continue
                    episode, frame, _ = schedule.frames.samples[i]
                    if stream in previous:
                        assert previous[stream] == (episode, frame["frame_index"] - 1)
                    previous[stream] = episode, frame["frame_index"]


def test_mixed_vst_counts_partial_group_and_parallel_gradients(model, collator):
    collator.ratio = 3
    frames = [sample(), sample("blue")]
    frames[1]["images"][0] = Image.new("RGB", (32, 64), "green")
    batch = collator(frames, 1, 2)
    assert batch["vst_valid"].sum(-1).tolist() == [2, 1]
    model.train()
    target = torch.tensor([0.2, 0.7])
    result = model(batch, target)
    result["loss"].backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert result["cache"].lengths.tolist() == [3]
    model.zero_grad(set_to_none=True)
    cache, loss = None, 0
    for frame, pt in zip(frames, target):
        step = model(collator([frame]), pt[None], cache)
        cache = step["cache"]
        loss = loss + step["loss"] / 2
    loss.backward()
    torch.testing.assert_close(loss, result["loss"], atol=1e-6, rtol=1e-6)
    for name, p in model.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(p.grad, expected[name], atol=2e-6, rtol=3e-5)


def test_trainable_checkpoint_roundtrip(model, collator):
    state = {n: p.clone() for n, p in trainable_state(model).items()}
    with torch.no_grad():
        expected = model(collator([sample()]))["bin_logits"]
        model.vst_embedding.add_(1)
        model.progress_head.weight.add_(0.5)
    load_trainable_state(model, state)
    with torch.no_grad():
        restored = model(collator([sample()]))["bin_logits"]
    torch.testing.assert_close(restored, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="names"):
        load_trainable_state(model, {})
    legacy = dict(state)
    legacy["qwen.model.language_model.layers.1.self_attn.vst_projections.o_proj.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="names"):
        load_trainable_state(model, legacy)
