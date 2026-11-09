"""供视觉编码器和进度模型共用的本地微型 Qwen3.5 权重。"""

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5ForConditionalGeneration


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(41)
    directory = tmp_path_factory.mktemp("tiny-qwen")
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
