"""覆盖时间轴、静止过滤、边界补帧及 M1 标签这几处容易改变训练含义的规则。"""

import json

import av
import numpy as np
import pytest
from PIL import Image

from mem_lite.datasets.m1_samples import build_sample, make_window, read_training_sample, sample_windows
from mem_lite.datasets.prepare_m1 import DEFAULT_CONFIG, prepare_m1
from mem_lite.datasets.video import extract_observations, iter_timed_frames, motion_fraction


@pytest.fixture
def config():
    return json.loads(DEFAULT_CONFIG.read_text())


@pytest.fixture
def video_path(tmp_path):
    """两秒 25 FPS 视频，只有第15帧发生一次局部变化，其余帧完全静止。"""
    path = tmp_path / "episode.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(50):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            if index >= 15:
                image[16:24, 24:32] = 255
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


@pytest.mark.parametrize(
    "case,left,right,expected,valid",
    [
        ("case1", 18, 10, [118, 200], 1),
        ("case1", 20, 10, None, 0),
        ("case2", 20, 20, [100, 200], 1),
        ("case3", 10, 18, [100, 182], 1),
        ("case3", 10, 20, None, 0),
        ("case4", 8, 10, [108, 190], 1),
        ("case4", 10, 10, None, 0),
    ],
)
def test_cases_and_strict_eighty_percent(case, left, right, expected, valid):
    result = make_window(100, 200, 300, case, left, right)
    assert result["target_interval"] == expected
    assert result["valid"] == valid


def test_half_up_rounding_and_video_edges():
    # 25帧的2%恰好半帧，四舍五入到1，不使用 Python round 的偶数舍入。
    result = make_window(0, 25, 25, "case1", 2, 10)
    assert result["offset_frames"] == [1, 3]
    assert result["window"] == [1, 25]
    assert result["coverage"] == 24 / 25
    assert make_window(0, 25, 25, "case2", 20, 20)["window"] == [0, 25]


def test_fixed_clock_for_25fps(video_path):
    candidates = [(index, time) for index, time, candidate, _ in iter_timed_frames(video_path, 10) if candidate]
    assert [index for index, _ in candidates] == [int(tick * 2.5) for tick in range(20)]
    assert all(time <= tick / 10 + 1e-9 for tick, (_, time) in enumerate(candidates))
    # 源帧率低于候选频率时，也不能重复同一原帧。
    assert len([i for i, _, candidate, _ in iter_timed_frames(video_path, 60) if candidate]) == 50


def test_static_filter_preserves_boundaries_and_clock(video_path, tmp_path, config):
    protected = {0, 1, 24, 25, 49}
    result = extract_observations(video_path, protected, tmp_path / "frames", config)
    assert protected <= set(result["kept_frames"])
    assert 15 in result["kept_frames"]  # 局部变化需要保留。
    assert len(result["kept_frames"]) < 10
    assert sum(row["candidate_10fps"] for row in result["observations"]) == 20
    assert any(not row["kept"] for row in result["observations"])
    assert result["timestamps"][49] == pytest.approx(49 / 25)


def test_slow_change_compares_to_last_kept():
    reference, _ = motion_fraction(Image.new("RGB", (64, 48), (0, 0, 0)), None, 64, 12)
    _, small_change = motion_fraction(Image.new("RGB", (64, 48), (8, 8, 8)), reference, 64, 12)
    _, accumulated = motion_fraction(Image.new("RGB", (64, 48), (16, 16, 16)), reference, 64, 12)
    assert small_change == 0
    assert accumulated == 1


def test_target_boundaries_are_sample_local():
    base = [0, 10, 20, 30, 40, 50]
    window = {**make_window(5, 45, 60, "case4", 10, 10), "stage_id": 0, "case": "case4", "draw": 0}
    sample = build_sample(window, base, [i / 25 for i in range(60)], "test task")
    indices = [frame["frame_index"] for frame in sample["inputs"]["frames"]]
    assert window["target_interval"] is None  # 恰好80%，无效，不补目标边界。
    assert json.loads(sample["target_text"]) == {"valid": 0}
    window.update(make_window(5, 45, 60, "case4", 8, 8))
    sample = build_sample(window, base, [i / 25 for i in range(60)], "test task")
    indices = [frame["frame_index"] for frame in sample["inputs"]["frames"]]
    assert indices == [8, 10, 20, 30, 40, 41]
    assert sample["target"] == {"valid": 1, "start_index": 0, "end_index": 6}
    assert base == [0, 10, 20, 30, 40, 50]


def test_empty_invalid_queue_is_skipped():
    window = {**make_window(0, 100, 100, "case1", 40, 0), "stage_id": 0, "case": "case1", "draw": 0}
    assert build_sample(window, [0], [i / 25 for i in range(100)], "task") is None


def test_end_to_end_samples(video_path, tmp_path, config):
    annotation = {
        "task_prompt": "点击按钮一次",
        "video_file": video_path.name,
        "fps": 25,
        "total_frames": 50,
        "frame_index_base": 0,
        "frame_range_convention": "[start_frame, end_frame)",
        "num_stages": 2,
        "stages": [
            {"stage_id": 0, "start_frame": 0, "end_frame": 25, "state_description": "尚未点击"},
            {"stage_id": 1, "start_frame": 25, "end_frame": 50, "state_description": "已点击一次"},
        ],
    }
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()
    annotation_path = annotation_dir / "episode.annotations.json"
    annotation_path.write_text(json.dumps(annotation, ensure_ascii=False))
    output = tmp_path / "prepared"
    summary = prepare_m1(annotation_path, output, config)
    assert summary["sample_count"] + summary["skipped_empty"] == 64
    skipped = json.loads((output / "skipped.json").read_text())
    assert skipped and all(window["valid"] == 0 for window in skipped)
    assert summary["by_case"]["case2"] == {"valid": 16, "invalid": 0}
    assert sample_windows(annotation, config) == sample_windows(annotation, config)
    samples = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    for sample in samples:
        frames, target = sample["inputs"]["frames"], sample["target"]
        assert all((output / frame["image"]).is_file() for frame in frames)
        assert [f["queue_index"] for f in frames] == list(range(len(frames)))
        if target["valid"]:
            first, end = sample["metadata"]["target_interval"]
            chosen = frames[target["start_index"] : target["end_index"]]
            assert chosen[0]["frame_index"] == first
            assert chosen[-1]["frame_index"] == end - 1
        else:
            assert target == {"valid": 0}
    batch_item = read_training_sample(samples[0], output)
    assert len(batch_item["images"]) == len(samples[0]["inputs"]["frames"])
    assert "点击按钮一次" in batch_item["prompt"]
    assert "尚未点击" not in batch_item["prompt"]
    assert batch_item["expert_tokens_path"] is None
    assert (output / "preview.html").is_file()
