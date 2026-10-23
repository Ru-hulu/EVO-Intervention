# ruff: noqa: RUF003
"""覆盖 5 Hz 帧网格、逐步阶段 / 槽位完成状态与回退标签。"""

import json
from pathlib import Path
import tempfile
import unittest

from qwen_training.build_episode_index import build_index, episode_labels, rollback_steps, step_frames
from qwen_training.build_sample_index import CAMERAS

NAMES = ["拿起套筒", "拿起螺钉", "将螺钉和套筒对齐并插入", "将装配好的组件放入盒子"]


def segment(nowid, restid, start, end):
    return {"nowid": str(nowid), "restid": restid, "name": NAMES[nowid - 1],
            "30FPSstart_frame": start, "30FPSend_frame": end}


# 1 → 2 → 3 → 回退重做 2（破坏 2、3）→ 3 → 4，每段 12 帧。
ROLLBACK = [segment(1, "2,3,4", 0, 11), segment(2, "3,4", 12, 23), segment(3, "4", 24, 35),
            segment(2, "3,4", 36, 47), segment(3, "4", 48, 59), segment(4, "", 60, 71)]


class EpisodeLabelTests(unittest.TestCase):
    def test_step_frames_use_global_grid(self):
        self.assertEqual(step_frames(0, 13), [0, 6, 12])
        self.assertEqual(step_frames(7, 20), [12, 18])
        self.assertEqual(step_frames(7, 10), [7])

    def test_slot_state_follows_segments_and_rollback(self):
        frames = step_frames(0, 71)
        stages, finished = episode_labels(ROLLBACK, {1, 2, 3, 4}, frames)
        self.assertEqual(stages, [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4])
        state = [[int(i in done) for i in (1, 2, 3, 4)] for done in finished]
        self.assertEqual(state[2], [1, 0, 0, 0])   # 段 2 内：只有 1 完成。
        self.assertEqual(state[4], [1, 1, 0, 0])   # 段 3 内：1、2 完成。
        self.assertEqual(state[6], [1, 0, 0, 0])   # 回退段首：2 变回 0。
        self.assertEqual(state[10], [1, 1, 1, 0])  # 最后一项在 episode 内保持 0。
        self.assertEqual(rollback_steps(state), [[6, [1]]])

    def test_frame_outside_episode_raises(self):
        with self.assertRaises(ValueError):
            episode_labels(ROLLBACK, {1, 2, 3, 4}, [72])

    def test_build_index_writes_episode_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recovery"
            (root / "labels").mkdir(parents=True)
            annotation = {"taskid": [{"id": str(i), "name": n} for i, n in enumerate(NAMES, 1)],
                          "tasks": [{"episodes": [{"episode": 0, "subtasks": ROLLBACK}]}]}
            (root / "labels" / "file-000.JSON").write_text(json.dumps(annotation), encoding="utf-8")
            (Path(tmp) / "file-000-delta.txt").write_text("1\n" * 72, encoding="utf-8")
            for name in CAMERAS.values():
                video = root / "videos" / name / "chunk-000" / "file-000.mp4"
                video.parent.mkdir(parents=True)
                video.touch()
            result = build_index(root, Path(tmp))
            record = json.loads((root / "episodes" / "episodes.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(result["summary"]["episodes_by_rollback_count"], {1: 1})
        self.assertEqual(record["stage"], [0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 3, 3])
        self.assertEqual([s["depends_on"] for s in record["steps"]], [[], [], [0, 1], [2]])
        self.assertEqual(len(record["slot_state"]), len(record["frame_indices"]))
        self.assertEqual(record["rollback_steps"], [[6, [1]]])
        self.assertTrue(all(0 <= p <= 1 for p in record["progress"]))


if __name__ == "__main__":
    unittest.main()
