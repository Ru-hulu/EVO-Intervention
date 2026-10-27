"""StreamForest 样本的三视角、历史时间和标签隔离检查。"""

import json
import shutil

from PIL import Image
import pytest
import torch

from mem_lite.datasets.m3_dataset import DATA_FORMAT, M3Dataset, Stage3Collator, VIEW_ORDER
from mem_lite.tests.test_m3_predictor import collator as collator


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / "episode"
    directory.mkdir()
    summary = {"format_version": DATA_FORMAT, "task_prompt": "按按钮三次", "history_dim": 16,
               "history_capacity": 8, "history_source": {"vision_sha256": "fixture", "method": "streamforest-v1"}}
    (directory / "summary.json").write_text(json.dumps(summary))
    for view, color in zip(VIEW_ORDER, ("red", "green", "blue"), strict=True):
        Image.new("RGB", (32, 32), color).save(directory / f"{view}.png")
    samples, previous = [], None
    for index, (frame, count, pt) in enumerate(((0, 0, 0.0), (3, 3, 0.6), (6, 5, 0.1), (7, 7, 1.0))):
        history = f"{frame}.pt" if count else None
        if history:
            torch.save({"tokens": torch.full((count, 16), float(index)), "through_frame": previous}, directory / history)
        samples.append({"frame_index": frame, "images": {view: f"{view}.png" for view in VIEW_ORDER},
                        "history": history, "pt": pt})
        previous = frame
    (directory / "samples.jsonl").write_text("\n".join(json.dumps(sample) for sample in samples) + "\n")
    return directory


def rewrite_sample(directory, index, **values):
    path = directory / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[index].update(values)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_views_history_lengths_and_nonmonotonic_pt(prepared):
    dataset = M3Dataset([prepared])
    first, boundary, regression, last = [dataset[i] for i in range(len(dataset))]
    assert (dataset.history_dim, dataset.history_capacity) == (16, 8)
    assert first["history_tokens"] is None
    assert [image.getpixel((0, 0)) for image in first["images"]] == [(255, 0, 0), (0, 128, 0), (0, 0, 255)]
    assert [len(row["history_tokens"]) for row in (boundary, regression, last)] == [3, 5, 7]
    assert regression["history_tokens"].eq(2).all()
    assert [row["target_pt"] for row in (first, boundary, regression, last)] == [0.0, 0.6, 0.1, 1.0]


@pytest.mark.parametrize("through_frame", [0, 6, 100])
def test_future_or_stale_history_is_rejected(prepared, through_frame):
    torch.save({"tokens": torch.ones(5, 16), "through_frame": through_frame}, prepared / "6.pt")
    with pytest.raises(ValueError, match="未来或过期"):
        M3Dataset([prepared])[2]


@pytest.mark.parametrize("pt", [float("nan"), float("inf"), -0.1, 1.1])
def test_bad_pt_is_not_clamped_or_inferred(prepared, pt):
    rewrite_sample(prepared, 1, pt=pt)
    with pytest.raises(ValueError, match="PT"):
        M3Dataset([prepared])[1]


def test_collator_separates_targets_and_pads_to_capacity(prepared, collator):
    dataset = M3Dataset([prepared])
    samples = [dataset[0], dataset[2]]
    batcher = Stage3Collator(collator.processor, history_dim=16, history_capacity=8)
    batch = batcher(samples)
    torch.testing.assert_close(batch["target_pt"], torch.tensor([0.0, 0.1]))
    assert batch["inputs"]["history_mask"].sum(1).tolist() == [0, 5]
    assert batch["inputs"]["history_valid_mask"].sum(1).tolist() == [0, 5]
    assert batch["inputs"]["history_tokens"].shape == (2, 8, 16)
    assert batch["inputs"]["image_grid_thw"].shape == (6, 3)
    changed = batcher([{**row, "target_pt": 0.99} for row in samples])
    assert "target_pt" not in batch["inputs"] and "labels" not in batch["inputs"]
    for key in batch["inputs"]:
        torch.testing.assert_close(batch["inputs"][key], changed["inputs"][key])


def test_missing_camera_never_uses_main_view_as_substitute(prepared):
    (prepared / "cam_right_wrist.png").unlink()
    with pytest.raises(FileNotFoundError):
        M3Dataset([prepared])[0]


def test_signature_relocation_and_episode_isolation(prepared, tmp_path):
    original = M3Dataset([prepared])
    relocated = tmp_path / "relocated"
    shutil.copytree(prepared, relocated)
    assert M3Dataset([relocated]).signature == original.signature
    rewrite_sample(relocated, 1, pt=0.3)
    assert M3Dataset([relocated]).signature != original.signature
    dataset = M3Dataset([prepared, relocated])
    assert dataset[0]["history_tokens"] is None and dataset[4]["history_tokens"] is None
    assert dataset[0]["episode_id"] != dataset[4]["episode_id"]


def test_online_dataset_does_not_open_history_files(prepared):
    for cache in prepared.glob("*.pt"):
        cache.unlink()
    dataset = M3Dataset([prepared], load_history=False)
    assert all(row["history_tokens"] is None for row in dataset)
