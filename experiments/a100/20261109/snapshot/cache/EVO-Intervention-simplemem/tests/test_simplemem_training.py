"""Training labels, frozen splits, unsupervised heads and checkpoint continuation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from simplememvla.model import SimpleMemForRewardPrediction
from simplememvla.training.data import RewardDataset, progress_at, sample_indices
from simplememvla.training.prepare import audit, progress_labels
from simplememvla.training.prepare_huawei import import_labels
from simplememvla.training.train import run
from tests import test_simplemem_reward as reward_tests
from tests.test_progress import tiny_backbone


class LabelTests(unittest.TestCase):
    def test_huawei_csv_import_and_recovery_source_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core, recovery, media = [root / name for name in ("core", "recovery", "media")]
            core.mkdir()
            sources = []
            for task, split, start, values, base in (
                ("Count/task", "train", 0, [0, 0.2, 0.8, 1], core),
                ("insert-screw-recover", "validation", 10, [0.2, 0.8, 0.1, 1], recovery),
            ):
                parent = f"{task}/file-000"
                unit_id = parent + ("/episode-01" if start else "")
                video = media / task / "30FPS/file-000.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(b"video")
                csv_path = base / "labels" / unit_id / "progress.csv"
                csv_path.parent.mkdir(parents=True)
                csv_path.write_text(
                    "frame_index,delta,pt\n" + "".join(f"{start + i},1,{value}\n" for i, value in enumerate(values))
                )
                source = dict(
                    task=task,
                    source_id=parent,
                    split=split,
                    frames=20 if start else 4,
                    fps=2,
                    videos=[f"/old/{task}/30FPS/file-000.mp4", None],
                    task_prompt="Insert",
                    pt_method="legacy-rollback-unchanged" if start else "whole-video-cumulative-visual-delta-v1",
                )
                metadata = dict(
                    source=source,
                    units=[
                        dict(
                            unit_id=unit_id,
                            frame_range=[start, start + 4],
                            progress_csv=str(csv_path),
                            progress_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                        )
                    ],
                )
                (base / "labels" / parent / "metadata.json").write_text(json.dumps(metadata))
                if not start:
                    sources.append(source)
            (core / "index.json").write_text(json.dumps(dict(sources=sources)))
            (core / "train.sample_files.json").write_text(json.dumps([str(core / "cache/train/Count/task/file-000")]))
            (core / "validation.sample_files.json").write_text("[]")
            report = import_labels(core, recovery, media, root / "out")
            self.assertEqual(report["splits"], {"train": 1, "val": 1})
            data = RewardDataset(root / "out/records.jsonl", media, "val", max_frames=4, width=32, height=32)
            with patch(
                "simplememvla.training.data.read_video", return_value=np.zeros((4, 32, 32, 3), dtype=np.uint8)
            ) as reader:
                feature = data[0]
            self.assertEqual(reader.call_args.args[1], [10, 11, 12, 13])
            self.assertEqual(feature["progress_targets"], [0.8, 1])
            self.assertEqual(progress_at(data.records[0]["progress"], [11, 12]), [0.8, 0.1])
            self.assertEqual(feature["success_targets"], [-100, -100])
            csv_path.write_text(csv_path.read_text().replace("0.8", "0.9"))
            with self.assertRaisesRegex(ValueError, "checksum"):
                import_labels(core, recovery, media, root / "out")

    def test_sampling_and_completion(self):
        ids = sample_indices(101, 30, 2, 6)
        self.assertEqual((ids[0], ids[-1], len(ids)), (0, 100, 6))
        self.assertEqual(sample_indices(1, 30, 2, 6), [0, 0])
        self.assertEqual(progress_at(None, [0, 1]), [-100, -100])
        self.assertEqual(progress_at({"ends": [2, 5]}, [0, 2, 3, 5]), [0, 0.5, 0.5, 1])
        annotation = dict(
            total_frames=6,
            frame_index_base=0,
            frame_range_convention="[start_frame, end_frame)",
            num_stages=2,
            stages=[
                dict(start_frame=0, end_frame=3, state_description="pick"),
                dict(start_frame=3, end_frame=6, state_description="place"),
            ],
        )
        self.assertEqual(progress_labels(annotation, "C07")["ends"], [2, 5])

    def test_frozen_split_and_missing_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "Count/test"
            (task / "annotations").mkdir(parents=True)
            (root / "manifest.json").write_text(
                json.dumps({"tasks": [dict(task_path="Count/test", train=["b"], val=["a"])]})
            )
            (task / "metadata.json").write_text(json.dumps(dict(task_id="C02", trajectory_index="metadata.jsonl")))
            annotation = dict(
                total_frames=10,
                fps=2,
                frame_index_base=0,
                frame_range_convention="[start_frame, end_frame)",
                num_stages=2,
                stages=[
                    dict(start_frame=0, end_frame=5, state_description="无阶段性事件"),
                    dict(start_frame=5, end_frame=10, state_description="已按蓝色按钮1次"),
                ],
            )
            (task / "annotations/a.json").write_text(json.dumps(annotation))
            rows = [
                dict(
                    sample_id=k,
                    source_instruction="Press",
                    total_frames=10,
                    fps=2,
                    has_annotation=k == "a",
                    annotation_path="annotations/a.json",
                    video_views={"main_view": {"video_path": k + ".mp4"}},
                )
                for k in ["a", "b"]
            ]
            (task / "metadata.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
            report = audit(root, root / "out", "test")
            record = json.loads((root / "out/records.jsonl").read_text())
            self.assertEqual(record["split"], "val")
            self.assertIsNone(record["success"])
            self.assertEqual(report["summary"]["eligible"], 1)
            self.assertIn("no_progress_or_success_supervision", report["excluded"][0]["reasons"])


class TinyDataset:
    def __init__(self, *args):
        self.records = [dict(progress={"kind": "test"}, success=None)] * 4

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return index


class TrainerTests(unittest.TestCase):
    def test_masked_heads_resume_and_full_backward(self):
        created = []

        def factory(path, config, **kwargs):
            backbone = tiny_backbone().cpu().float()
            backbone.config.video_token_id = 293
            model = SimpleMemForRewardPrediction(config, backbone)
            created.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return model

        def collate(features):
            batch = reward_tests.SimpleMemRewardTests().batch()
            batch["progress_targets"] = torch.tensor([[0.2, 0.8]])
            batch["success_targets"] = torch.tensor([[-100.0, -100.0]])
            return batch

        processor = SimpleNamespace(
            video_processor=SimpleNamespace(), tokenizer=SimpleNamespace(), save_pretrained=lambda path: None
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = root / "records.jsonl"
            records.write_text("{}\n")
            args = SimpleNamespace(
                seed=7,
                out=str(root / "run"),
                records=str(records),
                pairs=None,
                progress_only=True,
                media_root=str(root),
                fps=2,
                max_frames=4,
                width=32,
                height=32,
                views="all",
                allow_derived=True,
                model="tiny",
                device="cpu",
                mode="heads",
                lr=1e-3,
                backbone_lr=1e-3,
                resume=None,
                epochs=1,
                accumulate=2,
                max_steps=1,
                save_every=1,
                eval_samples=1,
            )
            with (
                patch("simplememvla.training.train.RewardDataset", TinyDataset),
                patch("simplememvla.training.train.AutoProcessor.from_pretrained", return_value=processor),
                patch("simplememvla.training.train.SimpleMemForRewardPrediction.from_backbone", side_effect=factory),
                patch("simplememvla.training.train.SimpleMemRewardCollator", return_value=collate),
            ):
                run(args)
                first = torch.load(root / "run/last.pt", weights_only=False)
                self.assertFalse(
                    torch.equal(first["model"]["progress_head.0.weight"], created[0]["progress_head.0.weight"])
                )
                for name in ("success_head.0.weight", "preference_head.0.weight"):
                    self.assertTrue(torch.equal(first["model"][name], created[0][name]))
                args.resume = str(root / "run/last.pt")
                args.max_steps = 2
                run(args)
                resumed = torch.load(root / "run/last.pt", weights_only=False)
                self.assertEqual((resumed["step"], resumed["position"]), (2, 4))
                args.resume = None
                args.out = str(root / "continuous")
                run(args)
                continuous = torch.load(root / "continuous/last.pt", weights_only=False)
                for key in resumed["model"]:
                    torch.testing.assert_close(resumed["model"][key], continuous["model"][key], rtol=0, atol=0)
                args.mode = "full"
                args.max_steps = 1
                args.out = str(root / "full")
                run(args)
                full = torch.load(root / "full/last.pt", weights_only=False)
                key = "backbone.model.language_model.embed_tokens.weight"
                self.assertFalse(torch.equal(full["model"][key], created[-1][key]))


if __name__ == "__main__":
    unittest.main()
