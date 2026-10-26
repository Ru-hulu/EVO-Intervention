"""M2 整数扰动、固定后继标签、帧去重、队列边界与确定性配对。"""

import json
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mem_lite.datasets.m2_dataset import M2Collator, M2Dataset, Stage2Dataset
from mem_lite.datasets.m2_samples import build_queue_samples, sample_event_window
from mem_lite.datasets.prepare_m2 import DEFAULT_CONFIG, prepare_m2
from mem_lite.tests.test_m1_data import video_path as video_path  # 复用合成视频 fixture


def stages_for(lengths):
    stages, start = [], 0
    for index, length in enumerate(lengths):
        stages.append({"stage_id": index, "start_frame": start, "end_frame": start + length,
                       "state_description": f"state-{index}"})
        start += length
    return stages


@pytest.mark.parametrize("lengths", [(100, 100, 100), (3, 1, 7, 2), (901, 17, 320, 6), (25, 39, 51)])
def test_integer_windows_preserve_all_constraints(lengths):
    stages, rng = stages_for(lengths), random.Random(42)
    saw_eighty = False
    for index, stage in enumerate(stages[:-1]):
        for _ in range(500):
            start, end = sample_event_window(stages, index, sum(lengths), rng)
            length = lengths[index]
            neighbor_lengths = lengths[max(0, index - 1):index + 2]
            assert 0 <= start < end <= sum(lengths)
            assert 5 * abs((end - start) - length) <= min(neighbor_lengths)
            overlap = min(end, stage["end_frame"]) - max(start, stage["start_frame"])
            assert 5 * overlap >= 4 * length
            saw_eighty |= 5 * overlap == 4 * length
            for neighbor in stages[max(0, index - 1):index] + stages[index + 1:index + 2]:
                overlap = max(0, min(end, neighbor["end_frame"]) - max(start, neighbor["start_frame"]))
                assert 5 * overlap <= neighbor["end_frame"] - neighbor["start_frame"]
    if lengths == (100, 100, 100):
        assert saw_eighty  # M2 接受80%，不能误套M1的严格大于规则。


def test_targets_stay_with_next_stage_even_when_event_ends_early():
    annotation = {"stages": stages_for((90, 108, 108, 34)), "total_frames": 340}
    base = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    windows = [[0, 85], [94, 195], [195, 303]]
    with patch("mem_lite.datasets.m2_samples.sample_event_window", side_effect=windows):
        sample = build_queue_samples(annotation, base, [i / 30 for i in range(340)], 1, 9)[0]
    assert [t["state_description"] for t in sample["targets"]] == ["state-1", "state-2", "state-3"]
    assert [t["prefix_length"] for t in sample["targets"]] == [1, 2, 3]
    assert len(sample["inputs"]["events"]) == 3
    for event, (start, end) in zip(sample["inputs"]["events"], windows, strict=True):
        frames = [frame["frame_index"] for frame in event["frames"]]
        assert frames == sorted({frame for frame in base if start <= frame < end} | {start, end - 1})
    assert 194 not in [frame["frame_index"] for frame in sample["inputs"]["events"][2]["frames"]]
    assert base[-1] == 330  # 不把扰动边界写回公共队列。


@pytest.fixture
def prepared(tmp_path, video_path):
    annotation = {
        "task_prompt": "task", "video_file": video_path.name, "total_frames": 50, "fps": 25,
        "frame_index_base": 0, "frame_range_convention": "[start_frame, end_frame)",
        "stages": stages_for((15, 20, 15)),
    }
    directory = tmp_path / "annotations"
    directory.mkdir()
    path = directory / "episode.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    config = json.loads(DEFAULT_CONFIG.read_text())
    output = tmp_path / "prepared"
    summary = prepare_m2(path, output, config)
    assert summary["queue_count"] == 8 and summary["prefix_count"] == 16
    return output


class ImageProcessor:
    def __call__(self, images, return_tensors):
        self.images = images
        return {"pixel_values": torch.zeros(len(images), 3), "image_grid_thw": torch.ones(len(images), 3)}


def test_prepare_load_and_collate_keep_video_queues_separate(prepared):
    dataset = M2Dataset([prepared])
    sample = dataset[0]
    record = dataset.samples[0][1]
    paths = {frame["image"] for event in record["inputs"]["events"] for frame in event["frames"]}
    assert len(sample["images"]) == len(paths)
    assert sample["state_descriptions"] == ["state-1", "state-2"]
    assert all((prepared / path).is_file() for path in paths)
    processor = SimpleNamespace(image_processor=ImageProcessor())
    batch = M2Collator(processor)([sample, sample])
    assert batch["queue_event_counts"] == [2, 2]
    assert batch["state_descriptions"] == ["state-1", "state-2", "state-1", "state-2"]
    assert batch["event_frame_indices"][:2] == sample["event_frame_indices"]
    expected = [[i + len(paths) for i in event] for event in sample["event_frame_indices"]]
    assert batch["event_frame_indices"][2:] == expected
    assert len(processor.image_processor.images) == 2 * len(paths)


def test_m1_pairing_is_reproducible_and_changes_with_epoch(prepared):
    sample = M2Dataset([prepared]).samples[0][1]
    frame = {**sample["inputs"]["events"][0]["frames"][0], "queue_index": 0}
    path = prepared / "m1.jsonl"
    records = [{"inputs": {"task_prompt": "task", "expert_tokens_path": None, "frames": [frame]},
                "target_text": str(index)} for index in range(30)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    dataset = Stage2Dataset([prepared], [path], seed=3)
    first = [dataset[i]["m1"]["response"] for i in range(len(dataset))]
    dataset.epoch = 1
    second = [dataset[i]["m1"]["response"] for i in range(len(dataset))]
    assert first != second
    dataset.epoch = 0
    assert first == [dataset[i]["m1"]["response"] for i in range(len(dataset))]


def test_single_stage_and_non_contiguous_annotations_are_rejected():
    annotation = {"stages": stages_for((10,)), "total_frames": 10}
    with pytest.raises(ValueError, match="至少需要两个"):
        build_queue_samples(annotation, [], range(10), 1, 0)
    annotation = {"stages": stages_for((10, 10)), "total_frames": 20}
    annotation["stages"][1]["start_frame"] += 1
    with pytest.raises(ValueError, match="连续"):
        build_queue_samples(annotation, [], range(20), 1, 0)
