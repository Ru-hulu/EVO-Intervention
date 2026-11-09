"""M3离线微型权重检查：三视角、历史插入、原生RoPE、分箱和梯度；不执行训练。"""

import math
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

from mem_lite.datasets.m3_inputs import M3Collator
from mem_lite.models.m3_predictor import M3ProgressPredictor, adjacent_bin_targets


@pytest.fixture
def model(checkpoint):
    return M3ProgressPredictor(
        checkpoint[0], history_dim=16, dtype=torch.float32,
        lora_rank=2, lora_alpha=4, lora_dropout=0, gradient_checkpointing=False,
    ).eval()


@pytest.fixture
def collator(checkpoint):
    from transformers import AutoTokenizer

    vision = checkpoint[1].config.vision_config
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            min_pixels=1024, max_pixels=4096, patch_size=vision.patch_size,
            temporal_patch_size=vision.temporal_patch_size, merge_size=vision.spatial_merge_size,
        ),
        video_processor=Qwen3VLVideoProcessor(),
        tokenizer=AutoTokenizer.from_pretrained(checkpoint[0], local_files_only=True),
    )
    processor.chat_template = processor.tokenizer.chat_template
    return M3Collator(processor, history_dim=16, history_capacity=128)


def sample(history=None, prompt="点击按钮三次"):
    return {"task_prompt": prompt, "history_tokens": history,
            "images": [Image.new("RGB", (32, 32), "red"), Image.new("RGB", (64, 32), "blue"),
                       Image.new("RGB", (32, 64), "green")]}


def test_three_views_and_mixed_history_slots_keep_order_and_mask(collator):
    history = torch.randn(128, 16)
    batch = collator([sample(history), sample()])
    assert batch["image_grid_thw"].shape == (6, 3)
    torch.testing.assert_close(batch["image_grid_thw"][:3], batch["image_grid_thw"][3:])
    assert batch["history_mask"].sum(1).tolist() == [128, 0]
    assert batch["attention_mask"].sum(1).diff().item() == -128
    assert batch["attention_mask"][batch["history_mask"]].eq(1).all()
    assert batch["mm_token_type_ids"][batch["history_mask"]].eq(0).all()
    mask = batch["history_mask"][0]
    first_image = (batch["input_ids"][0] == 6).nonzero()[0, 0]
    assert mask.nonzero()[-1, 0] + 1 == first_image
    # 移除历史槽后，其余token与无历史样本完全一致，三张图都保持原位置关系。
    stripped = batch["input_ids"][0][batch["attention_mask"][0].bool() & ~mask]
    plain = batch["input_ids"][1][batch["attention_mask"][1].bool()]
    torch.testing.assert_close(stripped, plain)
    torch.testing.assert_close(batch["history_tokens"][0], history)
    absent = collator([sample()])
    assert absent["history_tokens"].shape == (1, 128, 16)
    assert not absent["history_valid_mask"].any() and not absent["history_mask"].any()


def test_history_is_projected_continuously_and_positions_preserve_native_geometry(model, collator):
    history = torch.randn(128, 16)
    batch = collator([sample(history)])
    with patch.object(model.qwen.model, "forward", wraps=model.qwen.model.forward) as native, torch.no_grad():
        model(batch)
    inputs = native.call_args.kwargs
    torch.testing.assert_close(inputs["inputs_embeds"][batch["history_mask"]], model.history_projection(history))
    mask = batch["history_mask"][0]
    positions = inputs["position_ids"][:, 0, mask]
    assert torch.equal(positions[0], positions[1]) and torch.equal(positions[1], positions[2])
    assert positions.diff().eq(1).all()
    plain = collator([sample()])
    base_positions, _ = model.qwen.model.get_rope_index(
        plain["input_ids"], plain["mm_token_type_ids"], image_grid_thw=plain["image_grid_thw"],
        attention_mask=plain["attention_mask"],
    )
    # 每张图的空间RoPE几何不变，整体只被前置的128个历史位置平移。
    image = batch["mm_token_type_ids"][0] == 1
    plain_image = plain["mm_token_type_ids"][0] == 1
    torch.testing.assert_close(inputs["position_ids"][:, 0, image], base_positions[:, 0, plain_image] + 128)


def test_no_history_matches_original_native_multimodal_path_and_never_generates(model, collator):
    batch = collator([sample(), sample(prompt="依次完成三次不同的按钮点击，然后结束任务")])
    with torch.no_grad():
        reference = model.qwen.model(**{k: v for k, v in batch.items() if not k.startswith("history_")}, use_cache=False)
        expected = model.progress_head(reference.last_hidden_state[:, -1]).softmax(-1)
        with patch.object(model.qwen.lm_head, "forward", side_effect=AssertionError("不能使用词表头")), \
                patch.object(model.qwen, "generate", side_effect=AssertionError("不能生成文本")):
            result = model(batch)
    torch.testing.assert_close(result["bin_probs"], expected)
    assert result["bin_probs"].shape == (2, 101)
    torch.testing.assert_close(result["bin_probs"].sum(-1), torch.ones(2))
    torch.testing.assert_close(result["pt"], expected @ torch.linspace(0, 1, 101))
    assert ((result["pt"] >= 0) & (result["pt"] <= 1)).all()


def test_mixed_batch_matches_individual_and_uses_history(model, collator):
    samples = [sample(torch.randn(128, 16)), sample(prompt="把物品放入抽屉")]
    with torch.no_grad():
        mixed = model(collator(samples))["bin_logits"]
        singles = torch.cat([model(collator([item]))["bin_logits"] for item in samples])
        changed = model(collator([sample(torch.zeros(128, 16))]))["bin_logits"]
    torch.testing.assert_close(mixed, singles, atol=2e-6, rtol=2e-5)
    assert not torch.allclose(mixed[:1], changed, atol=1e-7, rtol=1e-6)


def test_training_scope_and_history_gradient_boundary(model, collator, checkpoint):
    model.train()
    model.qwen.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    history = torch.randn(128, 16, requires_grad=True)
    batch = collator([sample(history), sample()])
    result = model(batch, torch.tensor([0.2375, 1.0]))
    result["loss"].backward()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert all(name.startswith(("history_projection.", "progress_head.")) or
               (name.startswith("qwen.model.language_model.layers.") and "lora_" in name) for name in trainable)
    assert all(p.grad is not None and p.grad.isfinite().all() for p in trainable.values())
    assert any(p.grad.abs().sum() > 0 for name, p in trainable.items() if "lora_" in name)
    assert model.history_projection.weight.grad.abs().sum() > 0
    assert model.progress_head.weight.grad.abs().sum() > 0
    assert history.grad is None
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert not model.qwen.model.visual.training
    original = checkpoint[1].model.visual
    assert next(model.qwen.model.visual.parameters()).data_ptr() != next(original.parameters()).data_ptr()
    for key, value in model.qwen.model.visual.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key])


def test_soft_targets_endpoints_and_fractional_progress():
    values = torch.tensor([0.0, 0.2375, 0.5, 0.999, 1.0])
    targets = adjacent_bin_targets(values)
    torch.testing.assert_close(targets.sum(-1), torch.ones(5))
    assert targets[0, 0] == 1 and targets[-1, -1] == 1
    torch.testing.assert_close(targets[1, 23:25], torch.tensor([0.25, 0.75]))
    assert targets[2, 50] == 1
    torch.testing.assert_close(targets @ torch.linspace(0, 1, 101), values)
    assert ((targets > 0).sum(-1) <= 2).all()


def test_soft_ce_and_target_never_enters_predictor_inputs(model, collator):
    batch = collator([sample(torch.randn(128, 16)), sample()])
    with torch.no_grad():
        model.progress_head.weight.zero_()
        model.progress_head.bias.zero_()
        first = model(batch, torch.tensor([0.2375, 0.0]))
        second = model(batch, torch.tensor([1.0, 0.5]))
    assert first["num_samples"] == 2
    assert first["loss"].item() == pytest.approx(math.log(101))
    assert first["loss_sum"].item() == pytest.approx(2 * math.log(101))
    torch.testing.assert_close(first["pt"], torch.full((2,), 0.5))
    torch.testing.assert_close(first["bin_logits"], second["bin_logits"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_and_inference_share_precision_and_keep_progress_in_float32(checkpoint, collator, dtype):
    model = M3ProgressPredictor(checkpoint[0], history_dim=16, dtype=dtype,
                                lora_rank=2, lora_alpha=4, lora_dropout=0, gradient_checkpointing=False)
    batch = collator([sample(torch.randn(128, 16))])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(batch, torch.tensor([0.2375]))
    model.eval()
    with torch.no_grad():
        predicted = model(batch)
    torch.testing.assert_close(result["bin_logits"], predicted["bin_logits"], atol=0, rtol=0)
    assert result["pt"].dtype == result["bin_probs"].dtype == result["loss"].dtype == torch.float32
    expected = (result["bin_probs"].detach().double() * model.bin_centers.double()).sum(-1)
    torch.testing.assert_close(result["pt"].double(), expected, atol=1e-7, rtol=1e-7)
    result["loss"].backward()
    assert model.history_projection.weight.grad.isfinite().all()


def test_input_contract_rejects_missing_views_bad_history_or_truncation(collator):
    item = sample()
    item["images"].pop()
    with pytest.raises(ValueError, match="3个视角"):
        collator([item])
    with pytest.raises(ValueError, match="history_tokens"):
        collator([sample(torch.randn(129, 16))])
    collator.max_sequence_length = 64
    with pytest.raises(ValueError, match="不能静默截断"):
        collator([sample(torch.randn(128, 16))])


@pytest.mark.parametrize("target", [torch.tensor([-0.1]), torch.tensor([1.1]), torch.tensor([float("nan")])])
def test_invalid_progress_labels_are_not_silently_clamped(target):
    with pytest.raises(ValueError, match="target_pt"):
        adjacent_bin_targets(target)


def test_variable_history_lengths_ignore_all_padding_values(model, collator):
    items = [sample(torch.randn(3, 16)), sample(torch.randn(9, 16)), sample()]
    batch = collator(items)
    assert batch["history_tokens"].shape == (3, 128, 16)
    assert batch["history_valid_mask"].sum(1).tolist() == [3, 9, 0]
    with torch.no_grad():
        original = model(batch)["bin_logits"]
        individual = torch.cat([model(collator([item]))["bin_logits"] for item in items])
        batch["history_tokens"][~batch["history_valid_mask"]] = float("nan")
        padded = model(batch)["bin_logits"]
    torch.testing.assert_close(original, individual, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(original, padded, atol=0, rtol=0)
