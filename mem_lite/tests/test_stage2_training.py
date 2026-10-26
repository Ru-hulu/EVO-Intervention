"""第二阶段离线前向/梯度、checkpoint和归一化检查；不调用训练循环或optimizer.step。"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

from mem_lite.datasets.m2_dataset import Stage2Collator
from mem_lite.models.stage2_model import Stage2Model
from mem_lite.tests.test_state_decoder import checkpoint as checkpoint  # 本地微型权重 fixture
from mem_lite.train_stage2 import (
    load_trainable_state,
    normalize_gradients,
    read_config,
    save_checkpoint,
    trainable_state,
)


@pytest.fixture
def model(checkpoint):
    return Stage2Model(
        checkpoint[0], {"lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0, "gradient_checkpointing": False},
        {"hidden_dim": 16, "num_heads": 2}, {"hidden_dim": 16, "num_heads": 2},
        vision_batch_size=1, dtype=torch.float32,
    )


def memory_batch(model):
    vision = model.segmenter.qwen.config.vision_config
    width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    return {
        "pixel_values": torch.randn(16, width), "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 4], [1, 2, 2]]),
        "event_frame_indices": [[0], [0, 1], [2]], "queue_event_counts": [2, 1],
        "state_descriptions": ["已点击1次", "已点击2次", "已拿起瓶子"],
    }


def test_memory_loss_reaches_memory_modules_not_m1_and_keeps_prefixes(model):
    batch = memory_batch(model)
    model.train()
    with patch.object(model.compressor, "forward", wraps=model.compressor.forward) as compress:
        result = model.memory_loss(batch)
    queues = compress.call_args.args[0]
    assert [len(queue) for queue in queues] == [32, 64, 32]
    torch.testing.assert_close(queues[0], queues[1][:32])
    result["loss_sum"].backward()
    assert all(parameter.grad is None for parameter in model.segmenter.parameters())
    assert all(parameter.grad is None for parameter in model.decoder.language_model.parameters())
    for group in model.parameter_groups(1e-4, 1e-4)[1:]:
        assert all(p.grad is not None and p.grad.isfinite().all() for p in group["params"])
        assert all(p.grad.abs().sum() > 0 for p in group["params"])
    assert not model.segmenter.qwen.model.visual.training
    assert not model.decoder.language_model.training


def test_vision_chunking_preserves_loss(model):
    batch = memory_batch(model)
    with torch.no_grad():
        first = model.memory_loss(batch)
        model.vision_batch_size = 8
        second = model.memory_loss(batch)
    torch.testing.assert_close(first["loss_sum"], second["loss_sum"])


def test_joint_forward_and_disjoint_parameter_groups(model):
    # M1 这里使用真实语言前向的短文本输入；M2 同时使用真实视觉前向。
    batch = {"input_ids": torch.tensor([[8, 10, 12, 1]]), "attention_mask": torch.ones(1, 4, dtype=torch.long),
             "labels": torch.tensor([[-100, -100, 12, 1]])}
    result = model({"m1": batch, "memory": memory_batch(model)})
    (result["m1_loss_sum"] + result["memory_loss_sum"]).backward()
    groups = model.parameter_groups(2e-5, 1e-4)
    assert not {id(p) for p in groups[0]["params"]}.intersection(id(p) for p in groups[1]["params"])
    grouped = {id(p) for group in groups for p in group["params"]}
    assert grouped == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(p.grad is not None for group in groups for p in group["params"])
    assert result["m1_tokens"] == 2 and result["memory_tokens"] > 0


def test_native_processor_collates_both_branches_into_real_model(model):
    vision = model.segmenter.qwen.config.vision_config
    image_processor = Qwen2VLImageProcessor(
        min_pixels=1024, max_pixels=2048, patch_size=vision.patch_size,
        temporal_patch_size=vision.temporal_patch_size, merge_size=vision.spatial_merge_size,
    )
    processor = Qwen3VLProcessor(
        image_processor=image_processor, tokenizer=model.decoder.tokenizer,
        video_processor=Qwen3VLVideoProcessor(), chat_template=model.decoder.tokenizer.chat_template,
    )
    images = [Image.new("RGB", (32, 32), "red"), Image.new("RGB", (64, 32), "blue")]
    sample = {
        "m1": {"images": images, "prompt": "切分事件", "response": '{"valid": 1, "start_index": 0, "end_index": 2}'},
        "memory": {"images": images, "event_frame_indices": [[0], [0, 1]],
                   "state_descriptions": ["已拿起瓶子", "已完成1次撒调料"]},
    }
    batch = Stage2Collator(processor)([sample])
    assert batch["m1"]["mm_token_type_ids"].eq(1).sum() == 3
    with torch.no_grad():
        result = model(batch)
    assert result["m1_loss_sum"].isfinite() and result["memory_loss_sum"].isfinite()


def test_branch_token_normalization_matches_weighted_objective():
    # 模拟两rank对变长微批次的sum梯度做DDP平均；两分支token总数不同。
    first, second = torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(3.0))
    optimizer = torch.optim.AdamW([{"name": "m1", "params": [first]}, {"name": "memory", "params": [second]}])
    first.grad = torch.tensor((4.0 + 9.0) / 2)
    second.grad = torch.tensor((7.0 + 5.0) / 2)
    totals = torch.tensor([0.0, 5.0, 0.0, 8.0, 2.0, 6.0])
    normalize_gradients(optimizer, totals, 2, {"m1_loss_weight": 0.4, "memory_loss_weight": 1.2})
    assert first.grad.item() == pytest.approx(0.4 * 13 / 5)
    assert second.grad.item() == pytest.approx(1.2 * 12 / 8)


def test_stage2_checkpoint_restores_only_trainable_modules_and_cursor(model, tmp_path):
    optimizer = torch.optim.AdamW(model.parameter_groups(1e-4, 2e-4))
    # 人工设置两分支的优化器动量用于读写校验，不执行优化器更新。
    for group in optimizer.param_groups:
        parameter = group["params"][0]
        optimizer.state[parameter] = {
            "step": torch.tensor(7.0), "exp_avg": torch.full_like(parameter, 0.125),
            "exp_avg_sq": torch.full_like(parameter, 0.25),
        }
    before = {module: {key: value.clone() for key, value in values.items()}
              for module, values in trainable_state(model).items()}
    with patch("mem_lite.train_stage2.torch.cuda.get_rng_state", return_value=torch.tensor([1], dtype=torch.uint8)):
        save_checkpoint(model, optimizer, {"seed": 42}, tmp_path, epoch=1, next_batch=3, step=7)
    state = torch.load(tmp_path / "last.pt", weights_only=True)
    assert (state["epoch"], state["next_batch"], state["step"]) == (1, 3, 7)
    assert set(state["model"]) == {"m1_lora", "event_encoder", "m2_compressor", "history_projection"}
    assert all("lora_" in key for key in state["model"]["m1_lora"])
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    load_trainable_state(model, state["model"])
    optimizer.state.clear()
    optimizer.load_state_dict(state["optimizer"])
    for module, values in trainable_state(model).items():
        for key, value in values.items():
            torch.testing.assert_close(value, before[module][key])
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 2e-4]
    for group in optimizer.param_groups:
        restored = optimizer.state[group["params"][0]]
        assert restored["step"] == 7
        assert restored["exp_avg"].eq(0.125).all() and restored["exp_avg_sq"].eq(0.25).all()


def test_resume_keeps_config_and_does_not_require_old_m1_checkpoint(tmp_path):
    config = {"model_path": "/model", "output_dir": "/out", "m1_checkpoint": "/no-longer-needed.pt",
              "m1_sample_files": ["/m1"], "m2_sample_files": ["/m2"], "epochs": 1, "max_steps": None,
              "batch_size": 1, "gradient_accumulation_steps": 3, "vision_batch_size": 16,
              "log_every_steps": 1, "save_every_steps": 10, "m1_loss_weight": 1, "memory_loss_weight": 1}
    args = SimpleNamespace(model_path=None, output_dir=str(tmp_path), m1_checkpoint=None,
                           m1_sample_files=None, m2_sample_files=None, epochs=4, max_steps=90)
    updated = read_config(args, {"config": config})
    assert updated["epochs"] == 4 and updated["max_steps"] == 90
    assert updated["gradient_accumulation_steps"] == 3
    args.m1_checkpoint = "/different.pt"
    with pytest.raises(ValueError, match="不能再指定"):
        read_config(args, {"config": config})
