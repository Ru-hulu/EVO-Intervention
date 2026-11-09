# ruff: noqa: RUF001
"""检查累计状态边界、计数语义、视频读取与训练划分。"""
import json
import random
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from unittest import mock

import numpy as np

from qwen_training.datasets.memlong.labels import build, episode_labels
from qwen_training.datasets.memlong.tasks import TASKS, state_counts

TASK = '04-连续按按钮'


def annotation():
    return {'video_file': 'episode_000000.mp4', 'fps': 25, 'total_frames': 30,
            'frame_index_base': 0, 'frame_range_convention': '[start_frame, end_frame)', 'num_stages': 3,
            'stages': [{'stage_id': 0, 'start_frame': 0, 'end_frame': 10, 'state_description': '无阶段性事件'},
                       {'stage_id': 1, 'start_frame': 10, 'end_frame': 20, 'state_description': '已按蓝色按钮1次。'},
                       {'stage_id': 2, 'start_frame': 20, 'end_frame': 30, 'state_description': '已按蓝色按钮2次。'}]}


class LabelTests(unittest.TestCase):
    def test_completion_boundary_and_trailing_hold(self):
        result = episode_labels(annotation(), TASK, 25)
        frames = result['frames']
        self.assertEqual(frames[9]['counts'], [0])
        self.assertEqual(frames[10]['counts'], [1])
        self.assertEqual(frames[19]['counts'], [1])
        self.assertEqual(frames[20]['counts'], [2])
        self.assertEqual([f['progress'] for f in frames[20:]], [1] * 10)
        self.assertEqual([x['end'] for x in result['trace']], [10, 20])
        self.assertEqual(frames[0]['progress'], 0)
        self.assertTrue(all(a['progress'] <= b['progress'] for a, b in pairwise(frames)))

    def test_sampling_includes_last_video_frame(self):
        frames = episode_labels(annotation(), TASK)['frames']
        self.assertEqual([x['index'] for x in frames], [0, 5, 10, 15, 20, 25, 29])

    def test_rejects_gaps_and_count_jumps(self):
        for field, value in [('start_frame', 11), ('state_description', '已按蓝色按钮3次。')]:
            source = annotation()
            source['stages'][1][field] = value
            with self.assertRaises(ValueError):
                episode_labels(source, TASK)

    def test_rejects_missing_initial_state_and_count_overflow(self):
        source = annotation()
        source['stages'][0]['state_description'] = '已按蓝色按钮1次。'
        with self.assertRaises(ValueError):
            episode_labels(source, TASK)
        with self.assertRaises(ValueError):
            episode_labels(annotation(), TASK, 26)
        source = annotation()
        source['total_frames'] = 120
        source['num_stages'] = 12
        source['stages'] = [source['stages'][0]] + [
            {'stage_id': i, 'start_frame': i * 10, 'end_frame': (i + 1) * 10,
             'state_description': f'已按蓝色按钮{i}次。'} for i in range(1, 12)]
        with self.assertRaises(ValueError):
            episode_labels(source, TASK)

    def test_task_state_semantics(self):
        cases = [('01-按数字按键', '左键已按2次；中键已按7次；蓝键已按2次。', [2, 7, 2]),
                 ('03-输入数字序列', '已完成按键5次。', [5]),
                 ('06-连续敲鼓', '已拿起鼓槌、敲鼓5次并放回桌面。', [1, 5, 1]),
                 ('07-重复挤洗洁精', '已按压泵头5次；清洁布已放回桌面。', [1, 1, 5, 1]),
                 ('08-重复撒调料', '已完成5次撒调料动作；调料瓶已放回桌面。', [1, 5, 1])]
        for task, text, expected in cases:
            self.assertEqual(state_counts(task, text, 0), expected)

    def test_build_only_annotated_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'raw', Path(tmp) / 'labels'
            directory = root / TASK
            (directory / 'annotations').mkdir(parents=True)
            for i in range(8):
                (directory / f'episode_{i:06d}.mp4').touch()
                if i == 7:
                    continue
                source = annotation()
                source['video_file'] = f'episode_{i:06d}.mp4'
                source['task_prompt'] = '本条轨迹的秘密答案，不应该进入模型输入'
                (directory / 'annotations' / f'episode_{i:06d}.annotations.json').write_text(json.dumps(source))
            self.assertEqual(build(root, out, [TASK], inspect_video=lambda _: (30, 25)), 7)
            split = json.loads((out / '_splits.json').read_text())[TASK]
            groups = [split['pool'], [split['eval_demo']], split['train'], split['val']]
            ids = [episode for group in groups for episode in group]
            self.assertEqual(sorted(ids), list(range(7)))
            record = json.loads((out / TASK / 'episode0.json').read_text())
            self.assertNotIn('秘密答案', record['instruction'])
            self.assertEqual(record['cameras'], ['video'])
            with self.assertRaises(ValueError):
                build(root, out, [TASK], inspect_video=lambda _: (29, 25))


class VideoTests(unittest.TestCase):
    def test_mp4_rgb_order_duplicate_indices_and_bounds(self):
        import av

        from qwen_training.progress.data import read_frames

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'red.mp4'
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('mpeg4', rate=25)
                stream.width = stream.height = 32
                stream.pix_fmt = 'yuv420p'
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                image[..., 0] = 240
                for _ in range(3):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            frames = read_frames(path, [2, 0, 2], ['video'])
            self.assertEqual(len(frames), 3)
            self.assertEqual(len(frames[0]), 1)
            self.assertGreater(frames[0][0][..., 0].mean(), 200)
            self.assertLess(frames[0][0][..., 2].mean(), 20)
            np.testing.assert_array_equal(frames[0][0], frames[2][0])
            with self.assertRaises(ValueError):
                read_frames(path, [3])

    def test_training_uses_manifest_and_single_view(self):
        from qwen_training.progress import data
        from qwen_training.progress.config import Config

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / TASK
            directory.mkdir()
            for episode in range(4):
                record = {'task': TASK, 'episode': episode, 'instruction': TASKS[TASK][1],
                          'source': 'unused.mp4', 'cameras': ['video'], 'total_frames': 30,
                          **episode_labels(annotation(), TASK)}
                (directory / f'episode{episode}.json').write_text(json.dumps(record))
            (Path(tmp) / '_splits.json').write_text(json.dumps({TASK: {
                'pool': [0], 'eval_demo': 1, 'train': [2], 'val': [3]}}))
            def read(path, indices, cameras):
                self.assertEqual(cameras, ['video'])
                return [[np.zeros((16, 16, 3), dtype=np.uint8)] for _ in indices]
            with mock.patch.object(data, 'read_frames', read):
                for split, rng in [('train', random.Random(0)), ('val', None)]:
                    sample = data.RMBench(tmp, tmp, split).sample(0, Config(), rng)
                    self.assertEqual(sample['count'].shape[-1], 10)
                    self.assertEqual(sample['count'][-1].sum().item(), 2)
                    self.assertEqual(sample['progress'][-1].item(), 1)


if __name__ == '__main__':
    unittest.main()
