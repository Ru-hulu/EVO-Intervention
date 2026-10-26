"""离线单元检查：图像顺序、答案掩码、LoRA 范围与检查点，不运行训练。"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.nn import functional as F
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from mem_lite.datasets.m1_dataset import M1Collator, M1Dataset
from mem_lite.models.m1_segmenter import M1Segmenter
from mem_lite.train_m1 import read_config, save_checkpoint


class OfflineProcessor:
    """用可辨识的 token 检查拼接与 padding；不需要下载 tokenizer 或模型。"""

    def __init__(self):
        self.tokenizer = self
        self.pad_token_id, self.eos_token_id = 0, 2
        self.chats = []

    def encode(self, text, **kwargs):
        return [ord(char) + 3 for char in text]

    def apply_chat_template(self, chat, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        self.chats.append(chat)
        return chat[0]["content"]

    def __call__(self, text, images, **kwargs):
        self.images = images
        sequences = []
        for content in text:
            ids = [1]
            for item in content:
                ids.extend([250] if item["type"] == "image" else [10] * len(item["text"]))
            sequences.append(torch.tensor([*ids, 3]))
        width = max(len(ids) for ids in sequences)
        ids = torch.stack([F.pad(ids, (width - len(ids), 0)) for ids in sequences])
        return {
            "input_ids": ids, "attention_mask": ids.ne(0).long(),
            "mm_token_type_ids": ids.eq(250).long(),
            "pixel_values": torch.zeros(len(images), 4), "image_grid_thw": torch.ones(len(images), 3),
        }


def sample(image_count, response):
    return {
        "prompt": "切分事件", "images": [Image.new("RGB", (8, 8), (i * 30, 0, 0)) for i in range(image_count)],
        "response": response, "expert_tokens_path": "/unused/expert.pt",
    }


def test_collator_variable_queues_answer_only_and_expert_disabled():
    processor = OfflineProcessor()
    samples = [sample(3, '{"valid": 1, "start_index": 0, "end_index": 3}'), sample(1, '{"valid": 0}')]
    batch = M1Collator(processor)(samples)
    assert processor.images == samples[0]["images"] + samples[1]["images"]
    assert batch["image_grid_thw"].shape[0] == 4
    assert batch["mm_token_type_ids"].sum(dim=1).tolist() == [3, 1]
    assert batch["labels"][batch["attention_mask"] == 0].eq(-100).all()
    for row, item in enumerate(samples):
        supervised = batch["labels"][row].ne(-100)
        assert batch["labels"][row][supervised].tolist() == [*processor.encode(item["response"]), 2]
        assert batch["mm_token_type_ids"][row][supervised].eq(0).all()
        assert batch["input_ids"][row][supervised].tolist() == batch["labels"][row][supervised].tolist()
    rendered = str(processor.chats)
    assert "expert.pt" not in rendered and "end_index" not in rendered
    assert batch["labels"].shape == batch["input_ids"].shape == batch["mm_token_type_ids"].shape


def test_long_queue_is_not_silently_truncated():
    with pytest.raises(ValueError, match="不能截断队列"):
        M1Collator(OfflineProcessor(), max_sequence_length=8)([sample(4, '{"valid": 0}')])


def test_dataset_keeps_each_manifest_image_root_and_excludes_annotations(tmp_path):
    directories = []
    for i in range(2):
        directory = tmp_path / str(i)
        directory.mkdir()
        Image.new("RGB", (8, 8), (i * 100, 0, 0)).save(directory / "frame.png")
        record = {
            "inputs": {"task_prompt": "按按钮", "expert_tokens_path": None, "frames": [
                {"queue_index": 0, "frame_index": 100, "time_seconds": 4, "image": "frame.png"},
            ]},
            "target_text": '{"valid": 0}',
            "metadata": {"state_description": "不应该输入的状态标签", "coverage": 0.6},
        }
        (directory / "samples.jsonl").write_text(json.dumps(record), encoding="utf-8")
        directories.append(directory)
    dataset = M1Dataset([directories[0], directories[1] / "samples.jsonl"])
    assert len(dataset) == 2
    assert [dataset[i]["images"][0].getpixel((0, 0))[0] for i in range(2)] == [0, 100]
    assert "不应该输入的状态标签" not in dataset[0]["prompt"]
    assert "专家" not in dataset[0]["prompt"]
    assert "0@0.000s" in dataset[0]["prompt"]


@pytest.fixture
def segmenter():
    # 仅创建随机微型结构，检查真实 Qwen3.5/PEFT 的接入，不下载或训练权重。
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 256, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
        },
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 32, "num_position_embeddings": 16},
    )
    with patch("mem_lite.models.m1_segmenter.Qwen3_5ForConditionalGeneration.from_pretrained",
               return_value=Qwen3_5ForConditionalGeneration(config)):
        yield M1Segmenter("unused", dtype=torch.float32, lora_rank=2, lora_alpha=4, lora_dropout=0)


def test_lora_is_language_only_and_visual_stays_frozen(segmenter):
    segmenter.train()
    trainable = [name for name, parameter in segmenter.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith("qwen.model.language_model.layers.") and "lora_" in name for name in trainable)
    assert any("linear_attn" in name for name in trainable)
    assert any("self_attn" in name for name in trainable)
    assert all(parameter.dtype == torch.float32 for parameter in segmenter.parameters() if parameter.requires_grad)
    assert not segmenter.qwen.model.visual.training
    assert not any(parameter.requires_grad for parameter in segmenter.qwen.model.visual.parameters())
    assert not segmenter.qwen.lm_head.weight.requires_grad
    with pytest.raises(NotImplementedError, match="专家 tokens 尚未接入"):
        segmenter({}, expert_tokens=torch.zeros(1, 32))


def test_answer_loss_matches_full_causal_ce_without_prompt_logits(segmenter):
    # 固定隐藏向量检查损失计算，不执行 VLM 或反向传播。
    hidden = torch.randn(2, 6, 32)
    labels = torch.tensor([[-100, -100, -100, 8, 9, 2], [-100, -100, -100, -100, 7, 2]])
    batch = {"input_ids": torch.ones_like(labels), "attention_mask": torch.ones_like(labels), "labels": labels}
    with torch.no_grad(), patch.object(segmenter.qwen.model, "forward",
                                      return_value=SimpleNamespace(last_hidden_state=hidden)):
        result = segmenter(batch)
        full_logits = segmenter.qwen.lm_head(hidden)
        expected = F.cross_entropy(full_logits[:, :-1].reshape(-1, 256), labels[:, 1:].reshape(-1), reduction="sum")
    torch.testing.assert_close(result["loss_sum"], expected)
    assert result["text_tokens"].item() == 5
    assert torch.equal(batch["labels"], labels)


def test_frozen_visual_features_feed_event_encoder_without_language_model(segmenter):
    from mem_lite.models.event_encoder import EventEncoder, encode_event_intervals
    from mem_lite.models.event_queue import EventQueue

    # 两张小图像分别有 4、8 个原始 patch，视觉合并后各保留 1、2 个空间 tokens。
    vision = segmenter.qwen.config.vision_config
    patch_width = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    pixels = torch.randn(12, patch_width)
    grids = torch.tensor([[1, 2, 2], [1, 2, 4]])
    with patch.object(segmenter.qwen.model.language_model, "forward", side_effect=AssertionError("不应经过语言模型")):
        frames = segmenter.extract_frame_features(pixels, grids)
    assert [tuple(frame.shape) for frame in frames] == [(1, 32), (2, 32)]
    assert all(not frame.requires_grad and not torch.is_inference(frame) for frame in frames)
    encoder = EventEncoder(segmenter.visual_feature_dim, hidden_dim=16, num_heads=2)
    encoded = encode_event_intervals(encoder, frames, [(0, 2)])
    queue = EventQueue()
    queue.append(encoded[0])
    # 检查冻结 ViT 输出仍可用于训练可学习压缩模块，不更新任何权重。
    loss = (queue.as_tokens() * torch.randn(32, 16)).mean()
    loss.backward()
    assert encoder.input_projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in segmenter.parameters())


def test_checkpoint_adapter_and_cursor_roundtrip(segmenter, tmp_path):
    optimizer = torch.optim.AdamW([p for p in segmenter.parameters() if p.requires_grad])
    before = {key: value.clone() for key, value in get_peft_model_state_dict(segmenter.qwen).items()}
    with patch("mem_lite.train_m1.torch.cuda.get_rng_state", return_value=torch.tensor([1], dtype=torch.uint8)):
        save_checkpoint(
            segmenter, optimizer, {"epochs": 2}, tmp_path, epoch=1, next_batch=0, step=9, epoch_finished=True,
        )
    state = torch.load(tmp_path / "last.pt", weights_only=True)
    assert (tmp_path / "epoch-01.pt").is_file()
    assert (state["epoch"], state["next_batch"], state["step"]) == (1, 0, 9)
    assert all("lora_" in name for name in state["lora"])
    with torch.no_grad():
        for parameter in segmenter.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    set_peft_model_state_dict(segmenter.qwen, state["lora"])
    after = get_peft_model_state_dict(segmenter.qwen)
    for name in before:
        torch.testing.assert_close(before[name], after[name])


def test_resume_keeps_training_config_and_accepts_new_stop_target(tmp_path):
    args = SimpleNamespace(model_path=None, sample_files=None, output_dir=str(tmp_path), epochs=4, max_steps=20)
    old = {"model_path": str(tmp_path / "model"), "sample_files": [str(tmp_path / "samples.jsonl")],
           "batch_size": 2, "gradient_accumulation_steps": 3, "seed": 42, "epochs": 1, "max_steps": 5,
           "log_every_steps": 1, "save_every_steps": 10}
    config = read_config(args, {"config": old})
    assert config["epochs"] == 4 and config["max_steps"] == 20
    assert config["batch_size"] == 2 and config["gradient_accumulation_steps"] == 3
