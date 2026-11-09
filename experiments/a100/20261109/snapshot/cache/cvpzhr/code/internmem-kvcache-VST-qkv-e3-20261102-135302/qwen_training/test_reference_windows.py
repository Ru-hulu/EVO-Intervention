"""轻量检查：episode 边界、换序后的标签/三视角、NA 及索引生成。无模型依赖。"""

import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from qwen_training.build_reference_index import build_reference_index
from qwen_training.reference_io import format_frame_targets, make_reference_message, parse_frame_predictions
from qwen_training.reference_sampling import (
    make_frame_records, sample_episode_frames, split_reference_indices, swap_valid_blocks,
)


def window(indices, start=0, end=150):
    frames, targets = make_frame_records(indices, [i / (end - start) for i in range(end - start + 1)], start, end)
    return {"frames": frames, "targets": targets}


class ReferenceWindowTests(unittest.TestCase):
    def test_episode_boundary_padding_and_independent_intervals(self):
        rng = random.Random(41)
        self.assertEqual(sample_episode_frames(702, 1751, rng, current=702), [None] * 5 + [702])
        gaps = set()
        for _ in range(200):
            indices = sample_episode_frames(702, 1751, rng)
            valid = [i for i in indices if i is not None]
            self.assertTrue(all(702 <= i <= 1751 for i in valid))
            self.assertEqual(indices[:6-len(valid)], [None] * (6-len(valid)))
            self.assertTrue(all(15 <= b-a <= 30 for a, b in zip(valid, valid[1:])))
            gaps.update(b-a for a, b in zip(valid, valid[1:]))
        self.assertGreater(len(gaps), 1)

    def test_swap_moves_done_and_three_views_as_one_group(self):
        sample = window([0, 30, 60, 90, 120, 150])
        before = copy.deepcopy(sample)
        rng = random.Random(0)
        with patch.object(rng, "randint", return_value=3):
            swapped = swap_valid_blocks(sample, rng, 1)
        self.assertEqual(swapped["frame_order"], [3, 4, 5, 0, 1, 2])
        self.assertEqual([f["frame_index"] for f in swapped["frames"]], [90, 120, 150, 0, 30, 60])
        self.assertEqual([t["done"] for t in swapped["targets"]], [0, 0, 1, 0, 0, 0])
        self.assertEqual([i for i, f in enumerate(swapped["frames"]) if f["is_current"]], [2])
        self.assertEqual(sample, before)
        content = make_reference_message("装配", swapped["frames"])[0]["content"]
        self.assertEqual(sum(c["type"] == "image" for c in content), 13)
        text = "".join(c.get("text", "") for c in content)
        self.assertIn("frame-003（当前时刻）", text)
        self.assertNotIn("stage", text)

    def test_padding_stays_fixed_and_na_is_not_a_zero_label(self):
        sample = window([None, None, None, 90, 120, 150])
        swapped = swap_valid_blocks(sample, random.Random(5), 1)
        self.assertEqual(swapped["frame_order"][:3], [0, 1, 2])
        rows = parse_frame_predictions(format_frame_targets(swapped["targets"]))
        self.assertEqual(rows[:3], [{"progress": None, "done": None, "valid": False}] * 3)
        self.assertEqual([r["done"] for r in rows[3:]], [t["done"] for t in swapped["targets"][3:]])
        single = window([None] * 5 + [150])
        self.assertFalse(swap_valid_blocks(single, random.Random(1), 1)["order_shuffled"])

    def test_probability_gate_and_validation_identity(self):
        sample = window([0, 30, 60, 90, 120, 150])
        rng = random.Random(9)
        with patch.object(rng, "random", return_value=0.299):
            self.assertTrue(swap_valid_blocks(sample, rng)["order_shuffled"])
        with patch.object(rng, "random", return_value=0.301):
            self.assertFalse(swap_valid_blocks(sample, rng)["order_shuffled"])
        self.assertEqual(swap_valid_blocks(sample, rng, 0)["frames"], sample["frames"])

    def test_done_not_derived_from_rounded_progress(self):
        _, targets = make_frame_records([None] * 4 + [9, 10], [0.9999] * 10 + [1.0], 0, 10)
        lines = format_frame_targets(targets).splitlines()
        self.assertEqual(lines[-2:], ["progress=1.000 | done=0", "progress=1.000 | done=1"])
        bad = lines[:]; bad[-1] = "progress=1.100 | done=1"
        self.assertIsNone(parse_frame_predictions("\n".join(bad))[-1])
        self.assertEqual(parse_frame_predictions("\n".join(lines[:-1])), [None] * 6)

    def test_video_split_keeps_expert_source_out_of_validation(self):
        samples = [{"dataset": "normal", "video_id": f"file-{i:03d}",
                    "reference_source": {"dataset": "normal", "video_id": "file-000"}}
                   for i in range(10) for _ in range(3)]
        train, val = split_reference_indices(samples, 0.3, 1)
        train_videos = {samples[i]["video_id"] for i in train}
        val_videos = {samples[i]["video_id"] for i in val}
        self.assertFalse(train_videos & val_videos)
        self.assertIn("file-000", train_videos)
        self.assertEqual(len(val_videos), 3)

    def test_build_two_episodes_with_terminal_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("labels", "delta", "top", "left", "right", "refs"):
                (root / name).mkdir()
            # 只测试索引生成，此处视频是路径占位；解码由独立检查覆盖。
            for view in ("top", "left", "right"):
                (root / view / "file-000.mp4").touch()
            reference_images = []
            for i in range(5):
                p = root / "refs" / f"r{i}.png"; p.write_bytes(b"test-reference")
                reference_images.append(str(p))
            selection = root / "selection.json"
            selection.write_text(json.dumps({"reference_images": reference_images, "source_video": "file-000.mp4"}))
            annotation = {"taskid": [{"id": "1"}], "tasks": [{"episodes": [
                {"episode": i+1, "subtasks": [{"nowid": "1", "restid": "", "name": "装配",
                  "30FPSstart_frame": start, "30FPSend_frame": end}]}
                for i, (start, end) in enumerate(((12, 100), (101, 201)))
            ]}]}
            (root / "labels" / "file-000.JSON").write_text(json.dumps(annotation))
            (root / "delta" / "file-000-delta.txt").write_text("1 装配\n" * 202)
            result = build_reference_index(
                dataset_name="normal", label_dir=root/"labels", delta_dir=root/"delta",
                video_dirs={v: root/v for v in ("top", "left", "right")}, reference_selection=selection,
                output_dir=root/"out", samples_per_episode=4, terminal_samples=1, seed=3,
            )
            samples = [json.loads(line) for line in (root/"out/windows.jsonl").read_text().splitlines()]
            self.assertEqual((result["episodes"], result["windows"]), (2, 10))
            self.assertEqual(len((root/"out/episodes.jsonl").read_text().splitlines()), 2)
            for sample in samples:
                self.assertTrue(all(sample["episode_start_frame"] <= f["frame_index"] <= sample["episode_end_frame"]
                                    for f in sample["frames"] if f["valid"]))
                if sample["sample_type"] == "terminal":
                    self.assertEqual(sample["targets"][-1], {"valid": True, "progress": 1.0, "done": 1})
                self.assertTrue(all((root/"out"/path).is_file() for path in sample["reference_images"]))

    def test_image_loading_keeps_wrist_views_with_shuffled_frame(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        from qwen_training.reference_images import load_window_images
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = window([None, None, 60, 90, 120, 150])
            sample["reference_images"] = ["reference.png"] * 5
            Image.new("RGB", (8, 6), (1, 1, 1)).save(root/"reference.png")
            for i, frame in enumerate(sample["frames"]):
                if not frame["valid"]: continue
                frame["images"] = {}
                for view, color in (("top", (i+10, 0, 0)), ("left", (0, 100, 0)), ("right", (0, 0, 200))):
                    path = f"{i}-{view}.png"; Image.new("RGB", (8, 6), color).save(root/path)
                    frame["images"][view] = path
            swapped = swap_valid_blocks(sample, random.Random(3), 1)
            images = load_window_images(swapped, root)
            self.assertEqual(len(images), 13)
            self.assertEqual([image.getpixel((0, 0)) for image in images[5:7]], [(0, 0, 0)] * 2)
            current = next(i for i, frame in enumerate(swapped["frames"]) if frame["is_current"])
            self.assertEqual([image.getpixel((0, 0)) for image in images[5+current:8+current]],
                             [(15, 0, 0), (0, 100, 0), (0, 0, 200)])


if __name__ == "__main__":
    unittest.main()
