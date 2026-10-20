"""用本地随机小模型检查统一 CE 与生成后的进度解析，不下载模型。"""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from qwen_training.model import QwenProgressModel


class CharacterTokenizer:
    pad_token_id, eos_token_id = 0, 2

    def apply_chat_template(self, chat, **kwargs):
        prefix = [1] + [ord(c) + 3 for c in chat[0]["content"]] + [3]
        if chat[-1]["role"] == "assistant":
            return prefix + [ord(c) + 3 for c in chat[-1]["content"]] + [self.eos_token_id]
        return prefix

    def batch_decode(self, sequences, **kwargs):
        return ["".join(chr(i - 3) for i in ids.tolist() if i > 3) for ids in sequences]


class TextProgressModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = Qwen3_5Config(
            text_config=dict(vocab_size=256, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             head_dim=16, layer_types=["full_attention"],
                             rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                              "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                               out_hidden_size=8, num_position_embeddings=16),
        )
        qwen = Qwen3_5ForConditionalGeneration(config)
        with patch("qwen_training.model.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()), \
             patch("qwen_training.model.AutoImageProcessor.from_pretrained"), \
             patch("qwen_training.model.Qwen3_5ForConditionalGeneration.from_pretrained", return_value=qwen):
            self.model = QwenProgressModel("unused", dtype=torch.float32).eval()
        self.messages = [[{"role": "user", "content": "task"}], [{"role": "user", "content": "no task"}]]
        self.visual = torch.randn(2, 3, 8)
        self.mask = torch.tensor([[True, True, True], [True, False, False]])

    def test_all_answer_tokens_share_ce_and_gradients(self):
        responses = ["state\nprogress:0.637", "state"]  # Second sample has no task prompt.
        batch = self.model.build_inputs(self.messages, self.visual, self.mask, responses)
        for i, response in enumerate(responses):
            supervised = batch["labels"][i][batch["labels"][i] != -100].tolist()
            self.assertEqual(supervised, [ord(c) + 3 for c in response] + [2])
        output = self.model(self.messages, self.visual, self.mask, responses)
        singles = [self.model(self.messages[i:i+1], self.visual[i:i+1], self.mask[i:i+1], responses[i:i+1])
                   for i in range(2)]
        expected = sum(o.loss * o.text_tokens for o in singles) / sum(o.text_tokens for o in singles)
        torch.testing.assert_close(output.loss, expected)
        self.assertIs(output.loss, output.text_loss)
        self.assertFalse(hasattr(self.model, "progress_head"))
        output.loss.backward()
        gradient = self.model.visual_adapter[1].weight.grad
        self.assertTrue(gradient.isfinite().all() and gradient.abs().sum() > 0)

    def test_generate_reads_final_text_and_marks_missing_progress(self):
        replies = ["state\nprogress:0.637", "state"]
        ids = [torch.tensor([ord(c) + 3 for c in text] + [2]) for text in replies]
        generated = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True)
        with patch.object(self.model.qwen, "generate", return_value=generated):
            result = self.model.generate(self.messages, self.visual, self.mask)
        self.assertEqual(result["responses"], replies)
        self.assertAlmostEqual(result["progress"][0].item(), 0.637, places=6)
        self.assertEqual(result["progress_valid"].tolist(), [True, False])
        self.assertTrue(torch.isnan(result["progress"][1]))


if __name__ == "__main__":
    unittest.main()
