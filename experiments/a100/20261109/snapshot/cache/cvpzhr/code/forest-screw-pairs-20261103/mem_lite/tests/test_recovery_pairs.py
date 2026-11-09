"""Two-round supervision retains rollback, local time and source-video split boundaries."""

import csv
import json

import pytest

from mem_lite.datasets.prepare_recovery_pairs import paired_progress, prepare_index


def test_pair_progress_preserves_rollback_and_completed_first_round():
    result = paired_progress([0.0, 0.8, 0.2, 1.0], [0.0, 0.6, 0.1, 1.0], gap=2)
    assert result == pytest.approx([0.0, 0.4, 0.1, 0.5, 0.5, 0.5, 0.5, 0.8, 0.55, 1.0])
    # A failed second assembly does not erase the completed first assembly.
    assert min(result[6:]) >= 0.5


@pytest.mark.parametrize("first,second,gap", [([0, 0.9], [0, 1], 0), ([0, 1], [0, 1], -1),
                                              ([0, float("nan"), 1], [0, 1], 0)])
def test_invalid_single_round_labels_are_rejected(first, second, gap):
    with pytest.raises(ValueError):
        paired_progress(first, second, gap)


@pytest.fixture
def prepared_pairs(tmp_path, monkeypatch):
    root = tmp_path / "source"
    for directory in ("30FPS", "left_wrist", "right_wrist", "splitlable", "delta"):
        (root / directory).mkdir(parents=True)
    monkeypatch.setattr("mem_lite.datasets.prepare_recovery_pairs.probe_video",
                        lambda _: {"frames": 50, "fps": 30.0, "start_time": 0.0})
    for name in ("file-000", "file-001", "file-019"):
        for camera in ("30FPS", "left_wrist", "right_wrist"):
            (root / camera / f"{name}.mp4").write_bytes(b"fixture")
        episodes = []
        for ordinal in range(5):
            stages = [{"nowid": str(task), "restid": "2" if task == 1 else "",
                       "30FPSstart_frame": ordinal*10 + j*2,
                       "30FPSend_frame": ordinal*10 + j*2 + 1}
                      for j, task in enumerate((1, 2, 1, 2))]
            episodes.append({"episode": ordinal+1, "subtasks": stages})
        annotation = {"taskid": [{"id": "1"}, {"id": "2"}], "tasks": [{"episodes": episodes}]}
        (root / "splitlable" / f"{name}.JSON").write_text(json.dumps(annotation))
        (root / "delta" / f"{name}-delta.txt").write_text("1 ignored_description\n" * 50)
    output = tmp_path / "pairs"
    prepare_index(root, output, validation=("file-000",), review=("file-019",))
    return output


def test_index_pairs_overlap_only_within_the_same_source_split(prepared_pairs):
    output = prepared_pairs
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["episode_counts"] == {"train": 4, "validation": 4, "review": 4}
    records = [json.loads(line) for line in (output / "all.jsonl").read_text().splitlines()]
    for record in records:
        with (output / record["progress_csv"]).open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 18
        assert [int(r["frame_index"]) for r in rows] == list(range(18))
        source_start = record["source_frame_range"][0]
        assert [int(r["source_frame_index"]) for r in rows] == list(range(source_start, source_start+18))
        assert float(rows[7]["pt"]) == float(rows[8]["pt"]) == float(rows[9]["pt"]) == 0.5
        assert float(rows[10]["pt"]) == 0.625
        assert float(rows[-1]["pt"]) == 1.0
        assert (record["split"] == "review") == (record["source_video_id"] == "file-019")
    own = [r for r in records if r["source_video_id"] == "file-001"]
    assert [r["source_episodes"] for r in own] == [[1, 2], [2, 3], [3, 4], [4, 5]]
    assert {r["split"] for r in own} == {"train"}


def test_cache_validation_publishes_separate_lists_and_rejects_future_history(prepared_pairs, tmp_path):
    import torch
    from PIL import Image
    from mem_lite.datasets.m3_dataset import DATA_FORMAT, VIEW_ORDER
    from mem_lite.datasets.validate_recovery_pairs import validate

    records = [json.loads(line) for line in (prepared_pairs / "all.jsonl").read_text().splitlines()]
    reference = {"history_source": {"sample_fps": 5.0}, "model": {"history_dim": 2}, "history_capacity": 4}
    config = tmp_path / "reference.json"
    config.write_text(json.dumps(reference))
    cache = tmp_path / "cache"
    clips = {"num_clips": 36, "clips": [{"path": p} for r in records for p in r["videos"]]}
    (prepared_pairs / "clips-report.json").write_text(json.dumps(clips))
    for record in records:
        directory = cache / record["split"] / record["episode_id"]
        directory.mkdir(parents=True)
        with (prepared_pairs / record["progress_csv"]).open() as handle:
            labels = {int(r["frame_index"]): float(r["pt"]) for r in csv.DictReader(handle)}
        Image.new("RGB", (2, 2)).save(directory / "frame.png")
        samples, previous = [], None
        for frame in (0, 6, 12, 17):
            history = None if previous is None else f"{frame}.pt"
            if history:
                torch.save({"through_frame": previous, "tokens": torch.ones(2, 2)}, directory / history)
            samples.append({"frame_index": frame, "pt": labels[frame], "history": history,
                            "images": dict.fromkeys(VIEW_ORDER, "frame.png")})
            previous = frame
        (directory / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
        summary = {"format_version": DATA_FORMAT, "source": record, "task_prompt": record["task_prompt"],
                   "num_samples": 4, "history_dim": 2, "history_capacity": 4,
                   "history_source": reference["history_source"]}
        (directory / "summary.json").write_text(json.dumps(summary))
    result = validate(prepared_pairs, cache, config, workers=2)
    assert result["observations"] == 48 and result["history_tensors_checked"] == 36
    assert result["memory_continuous_across_internal_boundary"] and result["review_excluded"]
    for split in ("train", "validation", "review"):
        paths = json.loads((cache / f"{split}.sample_files.json").read_text())
        assert len(paths) == 4 and all(f"/{split}/" in p for p in paths)
    baseline = tmp_path / "baseline-episode"
    baseline.mkdir()
    (baseline / "summary.json").write_text(json.dumps({"source": {"task": "insert-screw-recover", "video_id": "file-000"}}))
    config.write_text(json.dumps({**reference, "sample_files": [str(baseline)]}))
    with pytest.raises(ValueError, match="used to train the initialization checkpoint"):
        validate(prepared_pairs, cache, config, workers=2)
    config.write_text(json.dumps(reference))
    corrupted = cache / "train" / "file-001" / "pair-AB" / "12.pt"
    torch.save({"through_frame": 17, "tokens": torch.ones(2, 2)}, corrupted)
    with pytest.raises(ValueError, match="non-causal history"):
        validate(prepared_pairs, cache, config, workers=2)
