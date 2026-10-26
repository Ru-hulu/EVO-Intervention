"""M3 数据接口离线检查：三视角顺序、逐帧标签、历史可用时间与输入隔离。"""

import json
import shutil

from PIL import Image
import pytest
import torch

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator, VIEW_ORDER, file_sha256
from mem_lite.tests.test_m3_predictor import collator as collator
from mem_lite.tests.test_state_decoder import checkpoint as checkpoint


@pytest.fixture
def prepared(tmp_path):
    directory = tmp_path / "episode"
    directory.mkdir()
    histories = torch.stack([torch.ones(128, 16), torch.full((128, 16), 2.0)])
    torch.save({"available_from_frame": [3, 7], "history_tokens": histories}, directory / "history.pt")
    summary = {"format_version": "mem-lite-m3-v1", "task_prompt": "按按钮三次", "history_dim": 16,
               "history_source": {"stage2_sha256": "fixture", "method": "fixture"},
               "history_sha256": file_sha256(directory / "history.pt")}
    (directory / "summary.json").write_text(json.dumps(summary))
    for view, color in zip(VIEW_ORDER, ("red", "green", "blue"), strict=True):
        Image.new("RGB", (32, 32), color).save(directory / f"{view}.png")
    samples = [{"frame_index": frame, "time_seconds": frame / 30,
                "images": {view: f"{view}.png" for view in VIEW_ORDER}, "history_index": history, "pt": pt}
               for frame, history, pt in ((0, None, 0.0), (3, 0, 0.6), (6, 0, 0.1), (7, 1, 1.0))]
    (directory / "samples.jsonl").write_text("\n".join(json.dumps(sample) for sample in samples) + "\n")
    return directory


def rewrite_sample(directory, index, **values):
    path = directory / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[index].update(values)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_three_views_and_latest_available_history_preserve_nonmonotonic_pt(prepared):
    dataset = M3Dataset([prepared], history_dim=16)
    assert len(dataset) == 4
    first, boundary, regression, last = [dataset[i] for i in range(len(dataset))]
    assert first["history_tokens"] is None
    assert [image.getpixel((0, 0)) for image in first["images"]] == [(255, 0, 0), (0, 128, 0), (0, 0, 255)]
    assert boundary["history_tokens"].eq(1).all() and regression["history_tokens"].eq(1).all()
    assert last["history_tokens"].eq(2).all()
    assert [row["target_pt"] for row in (first, boundary, regression, last)] == [0.0, 0.6, 0.1, 1.0]
    assert set(boundary) == {"images", "task_prompt", "history_tokens", "target_pt"}


@pytest.mark.parametrize("index,history", [(0, 0), (1, 1), (3, 0), (1, None)])
def test_future_or_stale_history_is_rejected(prepared, index, history):
    rewrite_sample(prepared, index, history_index=history)
    with pytest.raises(ValueError, match="最新已可用"):
        M3Dataset([prepared], history_dim=16)[index]


@pytest.mark.parametrize("pt", [float("nan"), float("inf"), -0.1, 1.1])
def test_bad_pt_is_not_clamped_or_inferred(prepared, pt):
    rewrite_sample(prepared, 1, pt=pt)
    with pytest.raises(ValueError, match="PT"):
        M3Dataset([prepared], history_dim=16)[1]


def test_collator_separates_targets_and_mixed_history(prepared, collator):
    dataset = M3Dataset([prepared], history_dim=16)
    samples = [dataset[0], dataset[2]]
    batch = Stage3Collator(collator.processor, history_dim=16)(samples)
    torch.testing.assert_close(batch["target_pt"], torch.tensor([0.0, 0.1]))
    assert batch["inputs"]["history_mask"].sum(1).tolist() == [0, 128]
    assert batch["inputs"]["image_grid_thw"].shape == (6, 3)
    changed = Stage3Collator(collator.processor, history_dim=16)([{**row, "target_pt": 0.99} for row in samples])
    assert "target_pt" not in batch["inputs"] and "labels" not in batch["inputs"]
    for key in batch["inputs"]:
        torch.testing.assert_close(batch["inputs"][key], changed["inputs"][key])


def test_missing_camera_never_uses_main_view_as_substitute(prepared):
    (prepared / "cam_right_wrist.png").unlink()
    with pytest.raises(FileNotFoundError):
        M3Dataset([prepared], history_dim=16)[0]


def test_signature_allows_relocation_but_detects_data_changes(prepared, tmp_path):
    original = M3Dataset([prepared], history_dim=16)
    relocated = tmp_path / "relocated"
    shutil.copytree(prepared, relocated)
    assert M3Dataset([relocated], history_dim=16).signature == original.signature
    rewrite_sample(relocated, 1, pt=0.3)
    assert M3Dataset([relocated], history_dim=16).signature != original.signature
    with (relocated / "history.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="构造记录"):
        M3Dataset([relocated], history_dim=16)


def test_empty_history_and_independent_episode_caches(prepared, tmp_path):
    empty = tmp_path / "empty-history"
    shutil.copytree(prepared, empty)
    torch.save({"available_from_frame": [], "history_tokens": torch.empty(0, 128, 16)}, empty / "history.pt")
    summary = json.loads((empty / "summary.json").read_text())
    summary["history_sha256"] = file_sha256(empty / "history.pt")
    (empty / "summary.json").write_text(json.dumps(summary))
    for index in range(4):
        rewrite_sample(empty, index, history_index=None)
    dataset = M3Dataset([prepared, empty], history_dim=16)
    assert dataset[3]["history_tokens"].eq(2).all()
    assert all(dataset[i]["history_tokens"] is None for i in range(4, 8))
