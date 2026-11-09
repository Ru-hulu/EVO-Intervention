import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import export_order as exporter


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "raw"
        dataset = self.source / "recording"
        (dataset / "meta/episodes/chunk-000").mkdir(parents=True)
        self.info = dict(codebase_version="v3.0", total_episodes=2, total_frames=20, fps=30,
                         features={c: {"dtype": "video"} for c in exporter.CAMERAS.values()},
                         video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        self.manifest = dict(format_reference="MINT-SJTU/ProPriMem", format_revision="test", tasks=[
            dict(task_id="O08", task_slug="test_sequence", task_name="Test sequence",
                 collection_task_id="local-id", collection_task_name="测试", sources=[
                     dict(dataset="recording", episodes=2)])])
        rows = []
        for i, (start, length) in enumerate([(0, 7), (7, 13)]):
            row = dict(episode_index=i, length=length, tasks=["Press one, then three"])
            for camera in exporter.CAMERAS.values():
                prefix = f"videos/{camera}"
                row.update({prefix + "/chunk_index": 0, prefix + "/file_index": 0,
                            prefix + "/from_timestamp": start / 30,
                            prefix + "/to_timestamp": (start + length) / 30})
            rows.append(row)
        exporter.write_json(dataset / "meta/info.json", self.info)
        pq.write_table(pa.Table.from_pylist(rows), dataset / "meta/episodes/chunk-000/file-000.parquet")
        for camera in exporter.CAMERAS.values():
            path = dataset / f"videos/{camera}/chunk-000/file-000.mp4"
            path.parent.mkdir(parents=True)
            with av.open(str(path), "w") as out:
                stream = out.add_stream("libx264", rate=30, options={"g": "12", "crf": "0"})
                stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
                for i in range(20):
                    pixels = np.full((24, 32, 3), (i * 11) % 256, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    frame.pts, frame.time_base = i, Fraction(1, 30)
                    for packet in stream.encode(frame):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)

    def test_exact_mid_gop_cut_and_unannotated_schema(self):
        plan = exporter.load_plan(self.manifest, self.source)
        output = self.root / "export"
        report = exporter.convert(plan, self.manifest, output)
        self.assertEqual((report["episodes"], report["videos"]), (2, 6))
        task = output / "Order/08-test_sequence"
        rows = [json.loads(line) for line in (task / "metadata.jsonl").read_text().splitlines()]
        self.assertEqual([r["total_frames"] for r in rows], [7, 13])
        for row in rows:
            self.assertFalse(row["has_annotation"])
            self.assertIsNone(row["annotation_path"])
            self.assertIsNone(row["task_prompt"])
            self.assertEqual(row["source_instruction"], "Press one, then three")
            self.assertEqual(row["suite"], "Order")
            for view in row["video_views"].values():
                path = task / view["video_path"]
                self.assertEqual(exporter.sha256(path), view["video_sha256"])
        # Independent pixel check of a non-keyframe-aligned second episode.
        source = self.source / "recording/videos/observation.images.left_wrist/chunk-000/file-000.mp4"
        def pixels(path):
            with av.open(str(path)) as inp:
                return [hashlib.sha256(f.to_ndarray(format="yuv420p").tobytes()).hexdigest()
                        for f in inp.decode(video=0)]
        self.assertEqual(pixels(source)[7:], pixels(task / rows[1]["video_views"]["left_view"]["video_path"]))
        self.assertEqual(list((task / "annotations").iterdir()), [])
        additions = json.loads((output / "manifest.additions.json").read_text())
        split = additions["tasks"][0]
        self.assertEqual(split["train"], [rows[0]["sample_id"]])
        self.assertEqual(split["val"], [rows[1]["sample_id"]])
        self.assertEqual(split["trajectory_index_sha256"], exporter.sha256(task / "metadata.jsonl"))
        with self.assertRaises(FileExistsError):
            exporter.convert(plan, self.manifest, output)

    def test_changed_count_and_duplicate_sources_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["tasks"][0]["sources"][0]["episodes"] = 3
        with self.assertRaisesRegex(ValueError, "count changed"):
            exporter.load_plan(manifest, self.source)
        manifest = copy.deepcopy(self.manifest)
        manifest["tasks"][0]["sources"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            exporter.load_plan(manifest, self.source)

    def test_truncated_video_does_not_publish_metadata(self):
        plan = exporter.load_plan(self.manifest, self.source)
        plan[0]["episodes"][1]["frames"] = 14
        output = self.root / "bad-export"
        with self.assertRaisesRegex(ValueError, "Truncated source"):
            exporter.convert(plan, self.manifest, output)
        self.assertEqual(json.loads((output / "export-status.json").read_text())["status"], "incomplete")
        self.assertFalse((output / "Order/08-test_sequence/metadata.jsonl").exists())

    def test_paths_cannot_escape_raw_root(self):
        with self.assertRaises(ValueError):
            exporter.child(self.source, "../outside")

    def test_second_shard_uses_its_own_timestamp_origin(self):
        import shutil

        dataset = self.source / "recording"
        path = dataset / "meta/episodes/chunk-000/file-000.parquet"
        rows = pq.read_table(path).to_pylist()
        for camera in exporter.CAMERAS.values():
            source = dataset / f"videos/{camera}/chunk-000/file-000.mp4"
            shutil.copyfile(source, source.with_name("file-001.mp4"))
            prefix = f"videos/{camera}"
            rows[1][prefix + "/file_index"] = 1
            rows[1][prefix + "/from_timestamp"] = 0.0
            rows[1][prefix + "/to_timestamp"] = 13 / 30
        pq.write_table(pa.Table.from_pylist(rows), path)
        plan = exporter.load_plan(self.manifest, self.source)
        output = self.root / "shard-export"
        exporter.convert(plan, self.manifest, output)
        records = json.loads((output / "provenance.json").read_text())["trajectories"]
        self.assertTrue(all(r["source_start_frame"] == 0 for r in records[3:]))


if __name__ == "__main__":
    unittest.main()
