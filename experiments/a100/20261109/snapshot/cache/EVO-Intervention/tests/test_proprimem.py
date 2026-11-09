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
