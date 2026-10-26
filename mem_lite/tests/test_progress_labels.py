"""事件等权、边界delta归属和真实视觉批次连续性的离线检查。"""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from mem_lite.datasets.prepare_progress import encode_images, event_equal_progress, load_vision_encoder
from mem_lite.tests.test_state_decoder import checkpoint as checkpoint


def annotation(boundaries=(0, 3, 5, 8)):
    return {"total_frames": boundaries[-1], "stages": [
        {"start_frame": a, "end_frame": b, "state_description": f"已完成{i}次"}
        for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]}


def test_all_intervals_equal_despite_different_visual_movement():
    # 三段变化量分别为10、100和2000；最后一段同样贡献1/3。
    delta = [0, 1, 3, 6, 25, 75, 1000, 1000]
    progress, events = event_equal_progress(delta, annotation())
    np.testing.assert_allclose(progress, np.array([0, 0.1, 0.4, 1, 1.25, 2, 2.5, 3]) / 3)
    assert [event["delta_sum"] for event in events] == [10, 100, 2000]
    assert [event["delta_frame_range_inclusive"] for event in events] == [[1, 3], [4, 5], [6, 7]]
    assert [event["end_frame_exclusive"] for event in events] == [3, 5, 8]


def test_initial_state_is_an_action_and_static_frames_do_not_gain_progress():
    progress, _ = event_equal_progress([0, 0, 0, 9, 0, 1, 0, 2], annotation())
    np.testing.assert_allclose(progress, np.array([0, 0, 0, 1, 1, 2, 2, 3]) / 3)


def test_one_frame_events_and_per_event_scale_invariance():
    labels = annotation((0, 1, 2, 4))
    progress, _ = event_equal_progress([0, 2, 8, 500], labels)
    np.testing.assert_array_equal(progress, [0, 1 / 3, 2 / 3, 1])
    first, _ = event_equal_progress([0, 1, 3, 6, 25, 75, 4, 6], annotation())
    scaled, _ = event_equal_progress([0, 10, 30, 60, 2.5, 7.5, 400, 600], annotation())
    np.testing.assert_allclose(first, scaled)


def test_single_interval_and_description_do_not_change_progress():
    labels = annotation((0, 4))
    labels["stages"][0]["state_description"] = "任务已完成"
    progress, events = event_equal_progress([0, 1, 2, 7], labels)
    np.testing.assert_allclose(progress, [0, 0.1, 0.3, 1])
    assert len(events) == 1 and events[0]["completion_frame"] == 3


@pytest.mark.parametrize("delta", [[0, 0, 0, 0, 1, 1, 0, 0], [0, 1e-10, 0, 0, 1, 1, 0, 0],
                                 [0, 1, 3, 6, 25, 75, 0, 0]])
def test_zero_or_tiny_events_require_review(delta):
    with pytest.raises(ValueError, match="需复核"):
        event_equal_progress(delta, annotation())


def test_last_interval_without_observable_transition_requires_review():
    with pytest.raises(ValueError, match="需复核"):
        event_equal_progress([0, 1, 1], annotation((0, 2, 3)))


def test_invalid_deltas_and_discontinuous_annotations_rejected():
    for bad in ([1] * 8, [0, -1, 1, 1, 1, 1, 0, 0], [0, float("nan"), 1, 1, 1, 1, 0, 0], [0, 1]):
        with pytest.raises(ValueError):
            event_equal_progress(bad, annotation())
    broken = annotation()
    broken["stages"][1]["start_frame"] = 4
    with pytest.raises(ValueError, match="连续"):
        event_equal_progress([0] + [1] * 7, broken)


def test_vision_only_loading_and_batch_independent_features(checkpoint):
    # 微型fixture缺少图像processor文件，补入同一视觉配置对应的原生processor。
    path, original = checkpoint
    vision = original.config.vision_config
    processor = Qwen2VLImageProcessor(
        min_pixels=4096, max_pixels=4096, patch_size=vision.patch_size,
        temporal_patch_size=vision.temporal_patch_size, merge_size=vision.spatial_merge_size,
    )
    processor.save_pretrained(path)
    model, processor, digest = load_vision_encoder(path, "cpu")
    assert len(digest) == 64 and not model.training
    assert not any(p.requires_grad for p in model.parameters())
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original.model.visual.state_dict()[key])
    images = [Image.new("RGB", (64, 64), color) for color in ("red", "blue", "red")]
    together = encode_images(images, model, processor, 4096)
    separate = torch.cat([encode_images([image], model, processor, 4096) for image in images])
    torch.testing.assert_close(together, separate, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(together[0], together[2])
    assert not torch.allclose(together[0], together[1])
    assert together.dtype == torch.float32 and not together.requires_grad


def test_batch_reuses_encoder_and_keeps_processing_after_invalid_video(tmp_path, monkeypatch):
    from mem_lite.datasets import prepare_progress as single
    from mem_lite.datasets import prepare_progress_batch as batch

    dataset = tmp_path / "dataset"
    annotations = dataset / "task" / "annotations"
    annotations.mkdir(parents=True)
    for i in range(3):
        video_file = f"episode_{i}.mp4"
        (annotations.parent / video_file).write_bytes(b"video fixture")
        label = {**annotation((0, 2, 4)), "task_prompt": "重复操作", "video_file": video_file,
                 "frame_index_base": 0, "frame_range_convention": "[start_frame, end_frame)"}
        (annotations / f"episode_{i}.annotations.json").write_text(json.dumps(label))
    (annotations.parent / "unannotated.mp4").write_bytes(b"must not process")
    encoder = (torch.nn.Linear(2, 2).requires_grad_(False), object(), "vision-hash")
    loads, extracted = [], []

    def load(model_path, device):
        loads.append((model_path, device))
        return encoder

    def extract(video, model, processor, batch_size, max_pixels):
        assert model is encoder[0] and processor is encoder[1]
        extracted.append(video.name)
        delta = [0, 1, 2, 0] if video.stem == "episode_1" else [0, 1, 2, 4]
        return np.array(delta), {"frame_count": 4, "fps": 30, "feature_dim": 2}

    monkeypatch.setattr(batch, "load_vision_encoder", load)
    monkeypatch.setattr(single, "extract_deltas", extract)
    output = tmp_path / "output"
    result = batch.prepare_progress_batch(dataset, tmp_path / "model", output, device="cpu")
    assert len(loads) == 1 and len(extracted) == 3
    assert result["status"] == "complete_with_errors"
    assert (result["completed"], result["failed"], result["completed_frames"]) == (2, 1, 8)
    assert json.loads((output / "result.json").read_text()) == result
    assert json.loads((output / "status.json").read_text()) == result
    records = [json.loads(line) for line in (output / "index.jsonl").read_text().splitlines()]
    assert [record["event_count"] for record in records] == [2, 2]
    assert records[-1]["video"].endswith("episode_2.mp4")
    assert not (output / "task/episode_1/episode_1.progress.csv").exists()
