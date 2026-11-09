# ruff: noqa: RUF001
"""Suite preprocessing semantics, split isolation and shared training-loader compatibility."""

import json
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from unittest import mock

import numpy as np

from qwen_training.datasets.proprimem.prepare import annotation_events, convert, prepare
from qwen_training.progress.config import Config
from qwen_training.progress.data import RMBench


def annotation(completed=True, repetitions=2):
    texts = ["无阶段性事件"] + [f"已按蓝色按钮{n}次。" for n in range(1, repetitions + 1)] if completed \
        else ["将 bowl 1 拿起并放到 plate 1（第1个子任务）", "全部子任务已完成"]
    return dict(total_frames=len(texts) * 10, fps=10, frame_index_base=0,
                frame_range_convention="[start_frame, end_frame)", num_stages=len(texts),
                stages=[dict(start_frame=i * 10, end_frame=(i + 1) * 10, state_description=t)
                        for i, t in enumerate(texts)])


class SuiteTests(unittest.TestCase):
    def test_current_and_completed_annotations_have_different_completion_boundaries(self):
        self.assertEqual([f for _, f in annotation_events(annotation(), ("blue_button", None, False))], [10, 20])
        self.assertEqual([f for _, f in annotation_events(annotation(False), ("libero", None, False))], [10])

    def test_progress_and_counts_are_monotonic_with_a_final_hold(self):
        row = dict(episode_index=1, sample_id="sample", source_instruction="Press the blue button.",
                   video_views={"main_view": dict(video_path="main.mp4")})
        record = convert(annotation(), row, dict(task_id="C02", suite="Count", task_name="Press"), Path("Count/task"))
        frames = record["frames"]
        self.assertEqual(frames[-1]["counts"], [2])
        self.assertEqual(frames[-1]["progress"], 1)
        self.assertEqual(frames[-1]["stage"], 1)  # none class, J=1
        self.assertEqual([d["frame"] for d in record["demo"]], [10])
        self.assertTrue(all(b["progress"] >= a["progress"] for a, b in pairwise(frames)))
        with self.assertRaisesRegex(ValueError, "capacity"):
            convert(annotation(repetitions=11), row, dict(task_id="C02", suite="Count", task_name="Press"), Path("task"))

    def test_source_segments_remain_segments_and_invalid_boundaries_are_rejected(self):
        a = annotation()
        for stage in a["stages"][1:]:
            n = stage["start_frame"] // 10
            stage["state_description"] = f"已完成（源动作段标注）：Clap hands.（{n}段）。"
        events = annotation_events(a, ("source_segments", None, False))
        self.assertEqual(events, [("Clap hands.", 10), ("Clap hands.", 20)])
        bad = annotation()
        bad["stages"][1]["start_frame"] += 1
        with self.assertRaisesRegex(ValueError, "contiguous"):
            convert(bad, {}, {}, Path("task"))

    def test_completed_counts_use_rmbench_steps_and_reject_multiple_changes(self):
        from qwen_training.datasets.rmbench.tasks import TASKS

        a = annotation()
        a["stages"][1]["state_description"] = "将两节电池放入电池槽（1次）"
        a["stages"][2]["state_description"] = "将两节电池放入电池槽（1次）；翻转一节电池尝试下一种方向组合（1次）"
        events = annotation_events(a, ("completed_counts", "battery_try", False))
        self.assertEqual(events, [(TASKS["battery_try"].steps[0], 10), (TASKS["battery_try"].steps[1], 20)])
        bad = annotation()
        bad["stages"][1]["state_description"] = "左键已按1次；中键已按1次"
        with self.assertRaisesRegex(ValueError, "exactly one completion"):
            annotation_events(bad, ("number_buttons", None, False))

    def test_different_step_definitions_never_share_demonstrations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "labels"
            task = root / "Count/03-drum"
            (task / "annotations").mkdir(parents=True)
            meta = dict(task_id="C03", suite="Count", task_name="Drum", source_collection="RoboValue",
                        trajectory_index="metadata.jsonl")
            (task / "metadata.json").write_text(json.dumps(meta))
            rows = []
            for episode in range(8):
                texts = ["无阶段性事件", "拿起鼓槌", "拿起鼓槌，敲鼓1次"]
                if episode >= 4:
                    texts.append("拿起鼓槌，敲鼓1次，放回桌面")
                a = annotation()
                a.update(total_frames=10 * len(texts), num_stages=len(texts),
                         stages=[dict(start_frame=i * 10, end_frame=(i + 1) * 10, state_description=text)
                                 for i, text in enumerate(texts)])
                name = f"annotations/{episode}.json"
                (task / name).write_text(json.dumps(a))
                rows.append(dict(episode_index=episode, sample_id=str(episode), has_annotation=True,
                                 annotation_path=name, source_instruction="Strike the drum.",
                                 video_views={"main_view": dict(video_path=f"{episode}.mp4")}))
            (task / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            _, splits = prepare(root, out)
            self.assertEqual(len(splits), 2)
            seen = []
            for key, split in splits.items():
                ids = set(split["train"] + split["pool"] + split["val"] + [split["eval_demo"]])
                seen.append(ids)
                definitions = [json.loads((out / key / f"episode{i}.json").read_text())["steps"] for i in ids]
                self.assertTrue(all(steps == definitions[0] for steps in definitions))
            self.assertEqual({frozenset(ids) for ids in seen}, {frozenset(range(4)), frozenset(range(4, 8))})

    def test_scripted_failures_keep_v4_progress_and_their_assigned_split(self):
        from qwen_training.progress_lite.data import Trajectories

        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "labels"
            task = root / "State/02-put_back_block-failures"
            (task / "annotations").mkdir(parents=True)
            meta = dict(task_id="S02", suite="State", task_name="Put a block back",
                        source_collection="RMBench-Failures", trajectory_index="metadata.jsonl")
            (task / "metadata.json").write_text(json.dumps(meta))
            # Rises to 1/3, freezes at 0 after an out-of-order step at frame 20.
            dense = [f / 30 for f in range(10)] + [1 / 3] * 10 + [0.0] * 20
            rows = []
            for episode in range(4):
                a = annotation()
                a.update(total_frames=40, num_stages=2,
                         stages=[dict(start_frame=0, end_frame=20, state_description="开始执行"),
                                 dict(start_frame=20, end_frame=40, state_description="越序")],
                         progress=dict(policy="retained_progress_v4", dense_progress=dense))
                name = f"annotations/{episode}.json"
                (task / name).write_text(json.dumps(a))
                rows.append(dict(episode_index=episode, sample_id=str(episode), has_annotation=True,
                                 annotation_path=name, source_instruction="Put the block back.",
                                 split="val" if episode == 3 else "train", task_success=False,
                                 video_views={"main_view": dict(video_path=f"{episode}.mp4")}))
            (task / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            _, splits = prepare(root, out)
            self.assertEqual(splits["State/02-put_back_block-failures"]["train"], [0, 1, 2])
            self.assertEqual(splits["State/02-put_back_block-failures"]["val"], [3])
            dataset = Trajectories(out, root, "train")
            with mock.patch("qwen_training.progress_lite.data.read_frames",
                            side_effect=lambda paths, indices, cameras, box: [[None] for _ in indices]):
                sample = dataset.sample(0, None)
            record = json.loads((out / "State/02-put_back_block-failures/episode0.json").read_text())
            expected = [dense[frame["index"]] for frame in record["frames"]]
            np.testing.assert_allclose(sample["progress"].numpy(), expected, atol=1e-6)
            self.assertEqual(float(sample["progress"][-1]), 0.0)

    def test_stable_disjoint_split_and_loader_use_task_local_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "labels"
            task = root / "Count/02-press"
            (task / "annotations").mkdir(parents=True)
            meta = dict(task_id="C02", suite="Count", task_name="Press", source_collection="RoboValue",
                        trajectory_index="metadata.jsonl")
            (task / "metadata.json").write_text(json.dumps(meta))
            rows = []
            for episode in range(8):
                name = f"annotations/{episode}.json"
                (task / name).write_text(json.dumps(annotation()))
                rows.append(dict(episode_index=episode, sample_id=str(episode), has_annotation=True,
                                 annotation_path=name, source_instruction="Press the blue button.",
                                 video_views={"main_view": dict(video_path=f"{episode}.mp4")}))
            rows.append(dict(episode_index=9, sample_id="missing", has_annotation=False))
            (task / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            report, splits = prepare(root, out)
            _, repeated = prepare(root, Path(tmp) / "again")
            self.assertEqual(splits, repeated)
            self.assertEqual(report["excluded"][0]["reason"], "missing_annotation")
            split = splits["Count/02-press"]
            groups = [set(split[n]) for n in ("train", "val", "pool")] + [{split["eval_demo"]}]
            self.assertEqual(sum(map(len, groups)), len(set.union(*groups)))
            self.assertEqual(set.union(*groups), set(range(8)))
            dataset = RMBench(out, root, "train", suites=["Count"])
            self.assertFalse(dataset.task_specs["Count/02-press"].rewind_safe)
            def read(path, indices, cameras):
                self.assertEqual(set(path), {"main_view"})
                return [[np.zeros((4, 4, 3), dtype=np.uint8)] for _ in indices]
            with mock.patch("qwen_training.progress.data.read_frames", side_effect=read):
                sample = dataset.sample(0, Config())
            self.assertEqual(sample["count"].shape[-2], 1)
            self.assertEqual(len(sample["frames"]), len(sample["progress"]))
            with self.assertRaises(FileExistsError):
                prepare(root, out)


if __name__ == "__main__":
    unittest.main()
