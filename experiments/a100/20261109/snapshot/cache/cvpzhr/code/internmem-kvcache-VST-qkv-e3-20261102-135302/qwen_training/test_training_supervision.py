"""覆盖联合采样、末尾进度文本和固定历史窗口的关键行为。"""

import copy
import unittest
from unittest.mock import patch

from qwen_training.build_sample_index import history_frames
from qwen_training.prompts import (
    INPUT_COMBINATIONS, TASK_INTENT, TASK_PROMPT, build_training_examples, describe_target, expected_stages,
)
from qwen_training.progress_text import append_progress, parse_progress


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
        self.assertEqual([parse_progress(s) for s in result["responses"]], [0.6, 0.6, 0.0, 0.0, None, None])
        for i, (mode, memory, _) in enumerate(INPUT_COMBINATIONS):
            prompt = result["messages"][i][0]["content"]
            self.assertEqual("先前长期记忆：" in prompt, memory)
            self.assertEqual("目标任务：" in prompt, mode != "missing")
            self.assertEqual(TASK_PROMPT in prompt, mode == "matched")
            self.assertTrue(result["responses"][i].startswith(describe_target(self.target, include_progress=False)))
            self.assertEqual("expect-set：" in result["responses"][i], mode == "matched")
            self.assertEqual("progress：" in result["responses"][i], mode != "missing")
            self.assertEqual("progress：" in prompt, mode != "missing")
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
        self.assertEqual(parse_progress(result["responses"][0]), 0.0)

    def test_seeded_sampling_and_causal_one_second_history(self):
        args = ([self.previous] * 100, [self.target] * 100)
        self.assertEqual(build_training_examples(*args, seed=12), build_training_examples(*args, seed=12))
        self.assertEqual(history_frames(240, 0), (90, 120, 150, 180, 210, 240))
        self.assertEqual(history_frames(75, 30), (None, None, None, None, 45, 75))


class ProgressTextTests(unittest.TestCase):
    def test_precision_bounds_and_optional_output(self):
        for value, expected in [(0.0, 0.0), (0.6374, 0.637), (1.0, 1.0)]:
            text = append_progress("任务状态。", value)
            self.assertEqual(parse_progress(text), expected)
            self.assertEqual(parse_progress(text.replace("：", ":") + "\n"), expected)
        self.assertEqual(append_progress("任务状态。", None), "任务状态。")

    def test_invalid_truncated_or_duplicate_is_not_zero(self):
        for text in ["", "任务有四个阶段", "progress：", "progress：0.6", "progress：1.001",
                     "progress：-0.100", "progress：nan", "progress：0.600\n补充文字",
                     "progress：0.300\nprogress：0.600"]:
            with self.subTest(text=text):
                self.assertIsNone(parse_progress(text))


if __name__ == "__main__":
    unittest.main()
