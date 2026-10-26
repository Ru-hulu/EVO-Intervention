"""微型本地权重检查：独立加载、答案掩码及文本损失的跨模块梯度，不执行训练。"""

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from torch.nn import functional as F
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration

from mem_lite.models.event_encoder import EventEncoder
from mem_lite.models.event_queue import EventQueue
from mem_lite.models.m1_segmenter import M1Segmenter
from mem_lite.models.m2_compressor import M2Compressor
from mem_lite.models.state_decoder import StateTextDecoder


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(41)
    directory = tmp_path_factory.mktemp("state-decoder-tiny-qwen")
    special = ["<pad>", "<|im_end|>", "<unk>", "<|im_start|>",
               "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    vocab = {token: index for index, token in enumerate(special + sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", eos_token="<|im_end|>",
        unk_token="<unk>", additional_special_tokens=special[3:],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' }}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}{{ '<|im_end|>\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    tokenizer.save_pretrained(directory)
    config = Qwen3_5Config(
        tie_word_embeddings=True, image_token_id=4, video_token_id=5, vision_start_token_id=6, vision_end_token_id=7,
        text_config={
            "vocab_size": len(vocab), "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
            "head_dim": 16, "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "tie_word_embeddings": True, "pad_token_id": 0, "eos_token_id": 1,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]},
        },
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "out_hidden_size": 32, "num_position_embeddings": 16},
    )
    original = Qwen3_5ForConditionalGeneration(config).eval()
    original.save_pretrained(directory)
    return directory, original


@pytest.fixture
def decoder(checkpoint):
    return StateTextDecoder(checkpoint[0], history_dim=16, dtype=torch.float32)


def test_loads_original_language_weights_without_vision_or_m1_adapter(decoder, checkpoint):
    original = checkpoint[1].state_dict()
    for key, value in decoder.language_model.state_dict().items():
        source = key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key
        torch.testing.assert_close(value, original[source], rtol=0, atol=0)
    segmenter = M1Segmenter(checkpoint[0], dtype=torch.float32, lora_rank=2, lora_alpha=4)
    assert any("lora_" in name for name, _ in segmenter.named_parameters())
    assert not any("visual" in name or "lora_" in name for name, _ in decoder.named_parameters())
    decoder_embedding = decoder.language_model.get_input_embeddings().weight
    assert decoder_embedding.data_ptr() != segmenter.qwen.get_input_embeddings().weight.data_ptr()
    assert decoder_embedding.data_ptr() == decoder.language_model.lm_head.weight.data_ptr()
    decoder.train()
    assert decoder.history_projection.training
    assert all(not module.training for module in decoder.language_model.modules())
    assert all(not parameter.requires_grad for parameter in decoder.language_model.parameters())
    assert [name for name, parameter in decoder.named_parameters() if parameter.requires_grad] == [
        "history_projection.weight", "history_projection.bias",
    ]


def test_continuous_history_answer_eos_and_padding_masks(decoder):
    histories = torch.randn(2, 128, 16)
    texts = ["已拿起瓶子。", "已拿起瓶子；已完成3次撒调料。"]
    batch = decoder.build_inputs(histories, texts)
    prefix_length = len(decoder.prefix_ids)
    prompt_length = prefix_length + 128 + len(decoder.suffix_ids)
    torch.testing.assert_close(
        batch["inputs_embeds"][:, prefix_length:prefix_length + 128], decoder.history_projection(histories),
    )
    assert batch["labels"][:, :prompt_length].eq(-100).all()
    assert batch["labels"][batch["attention_mask"] == 0].eq(-100).all()
    for row, text in enumerate(texts):
        expected = [*decoder.tokenizer.encode(text, add_special_tokens=False), decoder.tokenizer.eos_token_id]
        assert batch["labels"][row][batch["labels"][row] != -100].tolist() == expected
    # 标签改变不应改变任何提示或历史输入位置。
    other = decoder.build_inputs(histories, ["完全不同的标签", "另一个状态"])
    torch.testing.assert_close(batch["inputs_embeds"][:, :prompt_length], other["inputs_embeds"][:, :prompt_length])


def test_loss_matches_full_causal_ce_and_individual_examples(decoder):
    history = torch.randn(2, 128, 16)
    texts = ["一次", "三次，已经完成"]
    with torch.no_grad():
        actual = decoder(history, texts)
        batch = decoder.build_inputs(history, texts)
        labels = batch.pop("labels")
        logits = decoder.language_model(**batch, use_cache=False).logits
        expected = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), reduction="sum",
        )
        singles = [decoder(history[i:i + 1], [text]) for i, text in enumerate(texts)]
    torch.testing.assert_close(actual["loss_sum"], expected)
    torch.testing.assert_close(actual["loss_sum"], sum(item["loss_sum"] for item in singles))
    assert actual["text_tokens"] == sum(item["text_tokens"] for item in singles)


def test_text_gradient_reaches_all_memory_modules_but_not_decoder(decoder):
    encoder = EventEncoder(input_dim=8, hidden_dim=16, num_heads=2)
    compressor = M2Compressor(input_dim=16, hidden_dim=16, num_heads=2)
    encoded = encoder([[torch.randn(3, 8)], [torch.randn(2, 8), torch.randn(4, 8)]])
    queue = EventQueue()
    queue.append(encoded[0])
    first = queue.as_tokens()
    queue.append(encoded[1])
    history = compressor([first, queue.as_tokens()])
    result = decoder(history, ["已拿起瓶子", "已完成1次撒调料"])
    (result["loss_sum"] / result["text_tokens"]).backward()
    for module in (encoder, compressor, decoder.history_projection):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None and parameter.grad.isfinite().all(), name
            assert parameter.grad.abs().sum() > 0, name
    assert all(parameter.grad is None for parameter in decoder.language_model.parameters())


def test_length_limit_does_not_truncate_supervision(decoder):
    decoder.max_sequence_length = 128
    with pytest.raises(ValueError, match="不能静默截断"):
        decoder(torch.randn(1, 128, 16), ["已完成"])
