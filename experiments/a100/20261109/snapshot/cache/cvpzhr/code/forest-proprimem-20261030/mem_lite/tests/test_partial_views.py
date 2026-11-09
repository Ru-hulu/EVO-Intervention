"""副视角缺失时，缓存和在线入口必须产生相同输入、历史和预测。"""

import json

from PIL import Image
import pytest
import torch

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.datasets.prepare_stream import build_episode, iter_observations
from mem_lite.datasets.views import fill_views
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.runtime import StreamingProgressPredictor
from mem_lite.tests.test_m3_predictor import collator as collator
from mem_lite.tests.test_stream_pipeline import make_stream, origin as origin, write_video


@pytest.mark.parametrize("view_count", [1, 2, 3])
def test_partial_views_cache_matches_online(checkpoint, collator, origin, tmp_path, view_count):
    model = M3ProgressPredictor(checkpoint[0], dtype=torch.float32, lora_rank=2, lora_alpha=4,
                               lora_dropout=0, gradient_checkpointing=False).eval()
    observations = [{"frame_index": i*6, "pt": i/2,
                     "images": [Image.new("RGB", (32, 32), c)
                                for c in (color, "green", "blue")[:view_count]]}
                    for i, color in enumerate(("red", "blue", "green"))]
    stream = make_stream(model.qwen.model.visual, collator.processor, origin)
    directory = tmp_path / 'samples'
    build_episode(observations, stream, directory, 'task', 'episode', origin)
    dataset = M3Dataset([directory])
    batcher = Stage3Collator(collator.processor, dataset.history_dim, history_capacity=dataset.history_capacity)
    online = StreamingProgressPredictor(model, collator.processor, origin)
    for row, cached in zip(observations, dataset, strict=True):
        for i in range(view_count, 3):
            assert cached['images'][i].size == (64, 64)
            assert cached['images'][i].getbbox() is None
        with torch.no_grad():
            expected = model(batcher([cached])['inputs'])
            actual = online(row['images'], row['frame_index'], 'episode', 'task')
        torch.testing.assert_close(actual['pt'], expected['pt'])
    saved = [json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
    assert sum(v is None for v in saved[0]['images'].values()) == 3-view_count
    # 已存在的路径丢失仍报错；仅显式缺失槽位补图。
    (directory/saved[0]['images']['cam_high']).unlink()
    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_video_reader_keeps_real_wrist_and_pads_missing_slot(tmp_path):
    videos = [tmp_path/'main.mp4', tmp_path/'wrist.mp4']
    for p in videos:
        write_video(p, 'red', count=13, rate=30)
    rows = list(iter_observations(videos, {i:i/12 for i in range(13)}, end_frame=13))
    assert [r['frame_index'] for r in rows] == [0, 6, 12]
    assert all(r['images'][1] is not None and r['images'][2] is None for r in rows)
    assert fill_views([rows[0]['images'][0], None, rows[0]['images'][1]])[1].getbbox() is None
    with pytest.raises(ValueError, match='主视角'):
        fill_views([None, rows[0]['images'][1]])
