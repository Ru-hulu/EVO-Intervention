"""The exported dataset index must preserve frame order, RGB cameras and episode splits."""
import json
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from qwen_training.datasets.rmbench.labels import CAMERAS
from qwen_training.progress.data import RMBench, read_frames, read_video


def video(path, channel):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        stream.codec_context.thread_count = 1
        for t in range(4):
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[..., channel] = 60 + 40 * t
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class VideoDataTests(unittest.TestCase):
    def test_repeated_out_of_order_frames_and_camera_order(self):
        with tempfile.TemporaryDirectory() as td:
            paths = {camera: Path(td) / f"{camera}.mp4" for camera in CAMERAS}
            for channel, path in enumerate(paths.values()):
                video(path, channel)
            frames = read_frames(paths, [3, 0, 3, 1])
            self.assertEqual(len(frames), 4)
            for views, t in zip(frames, [3, 0, 3, 1], strict=True):
                self.assertEqual(len(views), 3)
                for channel, image in enumerate(views):
                    self.assertEqual(image.shape, (24, 32, 3))
                    self.assertEqual(int(image.mean((0, 1)).argmax()), channel)
                    self.assertAlmostEqual(float(image[..., channel].mean()), 60 + 40 * t, delta=8)
            head = read_frames(paths, [0], CAMERAS[:1])
            self.assertEqual(len(head[0]), 1)

    def test_single_video_empty_and_invalid_requests(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "video.mp4"
            video(path, 0)
            self.assertEqual(read_video(path, []), [])
            self.assertEqual(len(read_frames(path, [1])[0]), 1)
            for indices in ([-1], [4]):
                with self.assertRaises(ValueError):
                    read_video(path, indices)

    def test_index_filters_removed_tasks_without_changing_split(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            labels, data = root / "labels", root / "data"
            data.mkdir()
            for task in ("press_button", "observe_and_pickup"):
                (labels / task).mkdir(parents=True)
                for ep in (8, 9, 45):
                    (labels / task / f"episode{ep}.json").write_text("{}")
            rows = [dict(task_slug="press_button", episode_index=ep, video_views={
                "main_view": dict(camera=CAMERAS[0], video_path=f"episode{ep}.mp4")}) for ep in (8, 9, 45)]
            (data / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            self.assertEqual(RMBench(labels, data, "train").episodes, [("press_button", 9)])
            self.assertEqual(RMBench(labels, data, "val").episodes, [("press_button", 45)])
            self.assertEqual(RMBench(labels, data, "train").splits["press_button"]["eval_demo"], 8)

    def test_manifest_splits_and_standalone_video_sources_still_work(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "example").mkdir()
            (root / "example/episode1.json").write_text("{}")
            (root / "_splits.json").write_text(json.dumps({"example": {"train": [1]}}))
            dataset = RMBench(root, root, "train")
            self.assertEqual(dataset.episodes, [("example", 1)])
            self.assertEqual(dataset.source({"source": "example.mp4"}), root / "example.mp4")


if __name__ == "__main__":
    unittest.main()
