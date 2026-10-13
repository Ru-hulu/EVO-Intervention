"""只覆盖联合采样、监督掩码和固定历史窗口的关键行为。"""

import copy
import unittest
from unittest.mock import patch

from qwen_training.build_sample_index import history_frames
from qwen_training.prompts import (
    INPUT_COMBINATIONS, TASK_INTENT, TASK_PROMPT, build_training_examples, describe_target, expected_stages,
)

try:
    import torch
except ImportError:
    torch = None


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.previous = {"nowid": 2, "finishid": [1], "restid": [3, 4]}
        self.target = {"nowid": 3, "finishid": [1, 2], "restid": [4],
                       "intent": TASK_INTENT, "task_prompt": TASK_PROMPT, "progress": 0.6}

    def test_six_modes_and_unchanged_ground_truth(self):
        before = copy.deepcopy((self.previous, self.target))
        with patch("qwen_training.prompts.random.Random") as random_class:
            rng = random_class.return_value
            rng.choices.return_value = list(range(6))
            rng.choice.side_effect = lambda candidates: candidates[0]
            result = build_training_examples([self.previous] * 6, [self.target] * 6)
            self.assertEqual(rng.choices.call_args.kwargs["weights"], [49, 14, 14, 3, 14, 6])
        self.assertEqual(result["progress"], [0.6, 0.6, 0.0, 0.0, 0.6, 0.6])
        self.assertEqual(result["progress_mask"], [True, True, True, True, False, False])
        for i, (mode, memory, _) in enumerate(INPUT_COMBINATIONS):
            prompt = result["messages"][i][0]["content"]
            self.assertEqual("先前长期记忆：" in prompt, memory)
            self.assertEqual("目标任务：" in prompt, mode != "missing")
            self.assertEqual(TASK_PROMPT in prompt, mode == "matched")
            self.assertTrue(result["responses"][i].startswith(describe_target(self.target)))
            self.assertEqual("expect-set：" in result["responses"][i], mode == "matched")
        # 缺失 prompt 和记忆时没有具体阶段名称泄漏；只有通用输出要求。
        self.assertNotIn("拿起套筒", result["messages"][5][0]["content"])
        self.assertEqual((self.previous, self.target), before)

    def test_expect_set_uses_prerequisites_and_respects_unknown(self):
        wrong_order = {**self.target, "finishid": [2], "restid": [1, 4]}
        self.assertEqual(expected_stages(wrong_order), [1])  # 正在装配，不代表允许装配。
        self.assertEqual(expected_stages({**self.target, "finishid": []}), [1, 2])
        self.assertEqual(expected_stages(self.target), [3])
        self.assertEqual(expected_stages({**self.target, "finishid": [1, 2, 3]}), [4])
        self.assertIsNone(expected_stages({**self.target, "expect_set": None}))
        self.assertEqual(expected_stages({**self.target, "expect_set": []}), [])

    def test_mismatch_uses_its_own_optional_expect_set(self):
        target = {**self.target, "mismatched_prompts": [
            {"prompt": "打开抽屉", "expect_set": ["拉开抽屉"]},
        ]}
        with patch("qwen_training.prompts.random.Random") as random_class:
            rng = random_class.return_value
            rng.choices.return_value = [2]
            rng.choice.side_effect = lambda candidates: candidates[0]
            result = build_training_examples([self.previous], [target])
        self.assertIn("expect-set：拉开抽屉。", result["responses"][0])
        self.assertNotIn("expect-set：将螺钉", result["responses"][0])
        self.assertEqual(result["progress"], [0.0])

    def test_seeded_sampling_and_causal_one_second_history(self):
        args = ([self.previous] * 100, [self.target] * 100)
        self.assertEqual(build_training_examples(*args, seed=12), build_training_examples(*args, seed=12))
        self.assertEqual(history_frames(240, 0), (90, 120, 150, 180, 210, 240))
        self.assertEqual(history_frames(75, 30), (None, None, None, None, 45, 75))


@unittest.skipIf(torch is None, "需要 PyTorch 才能检查进度梯度")
class ProgressLossTests(unittest.TestCase):
    def test_mask_excludes_targets_and_gradients(self):
        from qwen_training.losses import masked_progress_mse
        prediction = torch.tensor([0.8, 0.3, 0.9], requires_grad=True)
        loss, count = masked_progress_mse(prediction, torch.tensor([0.6, 0.0, float("nan")]),
                                         torch.tensor([True, True, False]))
        self.assertEqual(count.item(), 2)
        self.assertAlmostEqual(loss.item(), 0.065, places=6)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0.2, 0.3, 0.0]))

    def test_all_missing_prompt_keeps_zero_gradient_path(self):
        from qwen_training.losses import masked_progress_mse
        prediction = torch.tensor([0.2, 0.7], requires_grad=True)
        loss, count = masked_progress_mse(prediction, torch.tensor([float("nan"), float("nan")]),
                                         torch.tensor([False, False]))
        self.assertEqual((loss.item(), count.item()), (0.0, 0))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
