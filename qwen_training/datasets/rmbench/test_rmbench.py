# ruff: noqa: RUF003
"""覆盖三类任务的实例切分、项对应、计数、进度、示范关键帧与目录输出。"""

import json
import tempfile
import unittest
from pathlib import Path

from qwen_training.datasets.rmbench.labels import MAX_SLOTS, build, episode_labels, instances, sample_frames
from qwen_training.datasets.rmbench.spec import FIXED, TaskSpec
from qwen_training.datasets.rmbench.tasks import TASKS

LEFT, MIDDLE = "Press the left button one time.", "Press the middle button one time."
CONFIRM = "Press the confirm button."
PRESS = [[LEFT, 10], [LEFT, 10], [MIDDLE, 10], [CONFIRM, 10]]
PUT_BACK = [["Pick up the block and move it to the center position.", 6],
            ["Pick up the block and move it to the center position.", 4], ["Press the button.", 10],
            ["Move the block back to the front mat.", 10]]
BATTERY = [["Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.",
             10],
           ["Pick up the left battery and place it into the battery slot in the negative direction.", 10],
           ["Pick up the right battery and place it into the battery slot in the negative direction.", 10]]
LEFT_TO_RIGHT = "Use right arm to pick up the left block and move it to the right tray."
SWAP = [["Use right arm to pick up the right block and move it to the middle tray.", 10], [LEFT_TO_RIGHT, 8],
        ["Use right arm to pick up the right block and move it to the right tray.", 2],
        ["Use left arm to pick up the left block and move it to the left tray.", 10], ["Press the button.", 10]]


class InstanceTests(unittest.TestCase):
    def test_classify_blocks_groups_four_segments(self):
        segments = [["Storage the first-class blocks.", 5]] * 8 + [["Storage the second-class blocks.", 5]] * 16
        spans = instances(segments, TASKS["classify_blocks"])
        self.assertEqual(len(spans), 6)
        self.assertEqual(spans[1][1:], (20, 39))

    def test_swap_blocks_merges_mislabeled_tail(self):
        spans = instances(SWAP, TASKS["swap_blocks"])
        self.assertEqual([span[1:] for span in spans], [(0, 9), (10, 19), (20, 29), (30, 39)])
        self.assertEqual(spans[1][0], LEFT_TO_RIGHT)

    def test_rejects_run_not_divisible_by_instance_size(self):
        with self.assertRaises(ValueError):
            instances([["Storage the first-class blocks.", 5]] * 3, TASKS["classify_blocks"])


class LabelTests(unittest.TestCase):
    def test_sample_frames_keep_last_frame(self):
        self.assertEqual(sample_frames(30, 25), [0, 5, 10, 15, 20, 25, 29])
        self.assertEqual(sample_frames(26, 25), [0, 5, 10, 15, 20, 25])

    def test_fixed_task_uses_generic_steps_and_order(self):
        labels = episode_labels(PUT_BACK, TASKS["put_back_block"], 25)
        self.assertNotIn("front", " ".join(labels["steps"]))   # 原垫子是记忆答案，不进输入。
        self.assertEqual([item["step"] for item in labels["trace"]], [0, 1, 2])
        self.assertEqual(labels["demo"], [{"step": 0, "frame": 9}, {"step": 1, "frame": 19}, {"step": 2, "frame": 29}])
        progress = [f["progress"] for f in labels["frames"]]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual((progress[0], progress[-1]), (0, 1))

    def test_fixed_task_rejects_wrong_instance_count(self):
        with self.assertRaises(ValueError):
            episode_labels(PUT_BACK[:3], TASKS["put_back_block"], 25)

    def test_scene_count_counts_per_step_and_demo_shows_each_kind_once(self):
        labels = episode_labels(PRESS, TASKS["press_button"], 25)
        self.assertEqual(labels["step_counts"], [2, 1, 1])
        self.assertEqual(labels["steps"], list(TASKS["press_button"].steps))   # 输入不含本条的次数。
        frames = {f["index"]: f for f in labels["frames"]}
        self.assertEqual(frames[10]["counts"], [1, 0, 0])
        self.assertEqual(frames[20]["counts"], [2, 0, 0])   # 第 2 次左键在第 19 帧结束。
        self.assertEqual(frames[20]["stage"], 1)
        self.assertEqual(frames[39]["counts"], [2, 1, 1])
        self.assertEqual(labels["demo"], [{"step": 0, "frame": 9}, {"step": 1, "frame": 29}, {"step": 2, "frame": 39}])

    def test_search_task_progress_uses_actual_attempts_and_demo_has_success(self):
        labels = episode_labels(BATTERY, TASKS["battery_try"], 25)
        self.assertEqual(labels["step_counts"], [1, 2])
        self.assertEqual(labels["max_counts"], [1, 3])
        frames = {f["index"]: f for f in labels["frames"]}
        self.assertAlmostEqual(frames[10]["progress"], 1 / 3, places=5)   # 共 3 次尝试，第 1 次在第 9 帧结束。
        progress = [f["progress"] for f in labels["frames"]]
        self.assertEqual(progress, sorted(progress))
        self.assertEqual((progress[0], progress[-1]), (0, 1))
        self.assertEqual(labels["demo"], [{"step": 0, "frame": 9}, {"step": 1, "frame": 19},
                                          {"step": None, "frame": 29}])

    def test_search_task_rejects_too_many_attempts(self):
        segments = [*BATTERY, BATTERY[1], BATTERY[2]]   # 共翻转 4 次，超过上限 3。
        with self.assertRaises(ValueError):
            episode_labels(segments, TASKS["battery_try"], 25)

    def test_rejects_more_than_max_slots(self):
        spec = TaskSpec(kind=FIXED, steps=tuple(str(i) for i in range(MAX_SLOTS + 1)), segments_per_instance=1)
        with self.assertRaises(ValueError):
            episode_labels([["x", 2]] * (MAX_SLOTS + 1), spec, 25)


class BuildTests(unittest.TestCase):
    def write_task(self, root, task, annotation):
        demo = root / task / "demo_clean"
        (demo / "instructions").mkdir(parents=True)
        (demo / "language_annotation.json").write_text(json.dumps(annotation))
        for key in annotation:
            episode = key.removeprefix("episode_")
            (demo / "instructions" / f"episode{episode}.json").write_text(json.dumps({"seen": ["do it"]}))

    def test_build_writes_one_json_per_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "labels"
            self.write_task(root, "press_button", {"episode_0": PRESS, "episode_3": PRESS})
            self.write_task(root, "put_back_block", {"episode_0": PUT_BACK})
            written = build(root, out, 25, tasks=["press_button"], count_frames=lambda _: 40)
            self.assertEqual(written, 2)
            record = json.loads((out / "press_button" / "episode3.json").read_text())
            self.assertEqual(record["source"], "press_button/demo_clean/data/episode3.hdf5")
            self.assertEqual((record["instruction"], record["kind"]), ("do it", "scene_count"))
            self.assertFalse((out / "put_back_block").exists())

    def test_build_rejects_frame_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            self.write_task(root, "press_button", {"episode_0": PRESS})
            with self.assertRaises(ValueError):
                build(root, Path(tmp) / "labels", 25, count_frames=lambda _: 41)


if __name__ == "__main__":
    unittest.main()
