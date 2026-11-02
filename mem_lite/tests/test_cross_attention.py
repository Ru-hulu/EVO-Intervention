"""External KV reads: gate learning, permutation invariance, isolation and placement."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3_5ForConditionalGeneration

from mem_lite.models.cross_attention import GatedHistoryAttention, HistoryDecoderLayer
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.runtime import load_checkpoint
from mem_lite.tests.test_m3_predictor import collator as collator, model as model, sample
from mem_lite.train_stage3 import DEFAULT_CONFIG, read_config


def test_zero_gate_is_identity_but_learns_to_open():
    torch.manual_seed(10)
    module = GatedHistoryAttention(32, 16, 16, 2)
    hidden = torch.randn(2, 5, 32, requires_grad=True)
    memory = torch.randn(2, 9, 16, requires_grad=True)
    valid = torch.tensor([[True]*9, [False]*9])
    queries = torch.ones(2, 5, dtype=torch.bool)
    result = module(hidden, memory, valid, queries)
    torch.testing.assert_close(result, hidden, atol=0, rtol=0)
    result.square().sum().backward()
    assert module.gate.grad.abs() > 0
    assert memory.grad is None
    assert module.k_proj.weight.grad.eq(0).all()
    with torch.no_grad():
        module.gate.fill_(0.3)
    module.zero_grad(set_to_none=True)
    changed = module(hidden, memory, valid, queries)
    assert not torch.allclose(changed[0], hidden[0])
    torch.testing.assert_close(changed[1], hidden[1], atol=0, rtol=0)
    changed.square().sum().backward()
    assert module.k_proj.weight.grad.abs().sum() > 0


def test_memory_permutation_preserves_output_and_padding_queries():
    torch.manual_seed(12)
    module = GatedHistoryAttention(32, 16, 16, 2)
    with torch.no_grad():
        module.gate.fill_(0.5)
    hidden, memory = torch.randn(1, 4, 32), torch.randn(1, 7, 16)
    valid = torch.tensor([[True, False, True, True, False, True, True]])
    queries = torch.tensor([[True, True, False, False]])
    result = module(hidden, memory, valid, queries)
    # Token contents retain their original spatiotemporal encoding when reordered.
    reordered = module(hidden, memory.flip(1), valid.flip(1), queries)
    torch.testing.assert_close(result, reordered)
    torch.testing.assert_close(result[:, 2:], hidden[:, 2:], atol=0, rtol=0)
    torch.testing.assert_close(module(hidden, memory[:, :0], valid[:, :0], queries), hidden)


def test_default_injection_after_layers_8_16_24(checkpoint, collator):
    config = copy.deepcopy(checkpoint[1].config)
    config.text_config.num_hidden_layers = 32
    config.text_config.layer_types = ["linear_attention"]*3 + ["full_attention"]
    config.text_config.layer_types *= 8
    native = Qwen3_5ForConditionalGeneration(config)
    with patch.object(Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=native):
        model = M3ProgressPredictor(checkpoint[0], history_dim=16, dtype=torch.float32,
                                    lora_rank=2, lora_alpha=4, gradient_checkpointing=False,
                                    cross_attention_dim=16, cross_attention_heads=2).eval()
    layers = model.qwen.model.language_model.layers
    assert [i+1 for i, layer in enumerate(layers) if isinstance(layer, HistoryDecoderLayer)] == [8, 16, 24]
    seen = []
    handles = [module.register_forward_hook(lambda m, args, out, n=n: seen.append(n))
               for n, module in model.memory_adapters.items()]
    try:
        with torch.no_grad():
            model(collator([sample(torch.randn(3, 16))]))
    finally:
        for handle in handles:
            handle.remove()
    assert seen == ["8", "16", "24"]


def test_checkpointing_matches_eager_across_multiple_pending_batches(model, collator):
    reference = copy.deepcopy(model).train()
    model.train()
    model.qwen.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    batches = [collator([sample(torch.randn(3, 16))]), collator([sample(torch.randn(9, 16))])]
    target = torch.tensor([0.23])
    # Both forwards happen before backward; a mutable last-memory hook would use the wrong batch.
    for m in (reference, model):
        losses = [m(batch, target)["loss"] for batch in batches]
        sum(losses).backward()
    for (name, param), (other_name, other) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        assert name == other_name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("old_format", ["mem-lite-streamforest-v1", "mem-lite-streamforest-crossattn-v1",
                                        "mem-lite-streamforest-crossattn-v2"])
def test_old_checkpoint_is_rejected_before_loading_model(tmp_path, old_format):
    path = tmp_path / "old.pt"
    torch.save({"format": old_format}, path)
    with pytest.raises(ValueError, match="v2"):
        load_checkpoint(path, device="cpu")


def test_config_reads_existing_sample_file_index(tmp_path):
    index = tmp_path / "train.sample_files.json"
    index.write_text(json.dumps([str(tmp_path / "video-a"), str(tmp_path / "video-b")]))
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(model_path="/model", sample_files=str(index))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = SimpleNamespace(config=str(path), model_path=None, sample_files=None, output_dir=None,
                           epochs=None, max_steps=None)
    loaded = read_config(args)
    assert loaded["sample_files"] == json.loads(index.read_text())
    assert loaded["model"]["cross_attention_layers"] == [8, 16, 24]
