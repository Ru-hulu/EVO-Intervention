"""恢复数据的回退监督、视频级隔离和单轮历史边界。"""

import csv
import json

import pytest

from mem_lite.datasets.prepare_recovery import build_caches, prepare_index
from mem_lite.tests.test_m3_predictor import collator as collator
from mem_lite.tests.test_stream_pipeline import write_video


@pytest.fixture
def recovery(tmp_path):
    root = tmp_path / "recovery"
    for name in ("30FPS", "left_wrist", "right_wrist", "splitlable", "delta"):
        (root / name).mkdir(parents=True)
    for video_id in ("file-000", "file-001", "file-002"):
        for view, color in zip(("30FPS", "left_wrist", "right_wrist"), ("red", "green", "blue")):
            write_video(root / view / f"{video_id}.mp4", color, count=72)
        episodes = []
        for number, start in enumerate((6, 42), 1):
            stages = [{"nowid": str(i), "restid": "2" if i == 1 else "",
                       "30FPSstart_frame": start + 6 * k, "30FPSend_frame": start + 6 * k + 5}
                      for k, i in enumerate((1, 2, 1, 2))]
            episodes.append({"episode": number, "subtasks": stages})
        annotation = {"taskid": [{"id": "1"}, {"id": "2"}], "tasks": [{"episodes": episodes}]}
        (root / "splitlable" / f"{video_id}.JSON").write_text(json.dumps(annotation))
        (root / "delta" / f"{video_id}-delta.txt").write_text("0 first\n" + "1 previous description\n" * 71)
    # 排除必须发生在读取标注/视角之前；不应让问题视频进入任何划分。
    (root / "splitlable/file-019.JSON").write_text("invalid excluded annotation")
    return root


def test_recovery_index_preserves_rollbacks_and_groups_whole_source_videos(recovery, tmp_path):
    output = tmp_path / "prepared"
    result = prepare_index(recovery, output, validation_videos=1)
    assert result["episode_counts"] == {"train": 4, "validation": 2}
    assert result["rollback_transitions"] == 6
    assert not set(result["splits"]["train"]) & set(result["splits"]["validation"])
    assert "file-019" not in result["splits"]["train"] + result["splits"]["validation"]
    repeated = prepare_index(recovery, tmp_path / "same-seed", validation_videos=1)
    assert repeated["splits"] == result["splits"]
    rows = [json.loads(line) for split in ("train", "validation")
            for line in (output / f"{split}.jsonl").read_text().splitlines()]
    for row in rows:
        with (output / row["progress_csv"]).open() as source:
            values = list(csv.DictReader(source))
        start, end = row["source_frame_range"]
        assert [int(value["frame_index"]) for value in values] == list(range(start, end))
        progress = [float(value["pt"]) for value in values]
        assert progress[:12] == pytest.approx([(i + 1) / 12 for i in range(12)])
        assert progress[12] == pytest.approx(1 / 12)  # 失败时清掉需要重做的任务贡献。
        assert progress[-1] == 1.0


def test_recovery_missing_real_view_is_rejected(recovery, tmp_path):
    (recovery / "left_wrist/file-001.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="真实视角"):
        prepare_index(recovery, tmp_path / "prepared", validation_videos=1)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("count,rate", [(71, 30), (72, 25)])
def test_recovery_misaligned_views_are_rejected(recovery, tmp_path, count, rate):
    write_video(recovery / "right_wrist/file-002.mp4", "blue", count=count, rate=rate)
    with pytest.raises(ValueError, match="三视角"):
        prepare_index(recovery, tmp_path / "prepared", validation_videos=1)


def test_recovery_cache_uses_only_one_round_and_resets_memory(
    recovery, tmp_path, checkpoint, collator,
):
    from mem_lite.datasets.m3_dataset import M3Dataset

    prepared, cache = tmp_path / "prepared", tmp_path / "cache"
    prepare_index(recovery, prepared, validation_videos=1)
    collator.processor.save_pretrained(checkpoint[0])
    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps({"st_memory_windows": [1, 2], "st_memory_tokens": [4, 2],
                                 "event_split_window": 2, "long_memory_tokens_per_frame": 1,
                                 "long_memory_tokens_quota": 4}))
    outputs = build_caches(prepared, checkpoint[0], cache, device="cpu", max_pixels=4096,
                           min_pixels=1024, memory_config=memory)
    for split, paths in outputs.items():
        for path in paths:
            dataset = M3Dataset([path])
            source = dataset.episodes[0][1]["source"]
            start, end = source["source_frame_range"]
            assert [row["frame_index"] for row in dataset] == [start, start + 6, start + 12, start + 18, end - 1]
            assert dataset[0]["history_tokens"] is None
            assert dataset[1]["history_tokens"] is not None
            assert dataset[-1]["target_pt"] == 1.0
            assert dataset[2]["target_pt"] < dataset[1]["target_pt"]
            assert source["split"] == split
            assert "file-019" not in source["episode_id"]
    config = json.loads((cache / "train.config.json").read_text())
    assert config["sample_files"] == outputs["train"]
    assert set(outputs["train"]).isdisjoint(outputs["validation"])


def test_recovery_cache_rejects_modified_split_or_labels(recovery, tmp_path):
    prepared = tmp_path / "prepared"
    manifest = prepare_index(recovery, prepared, validation_videos=1)
    manifest["splits"]["validation"].append(manifest["splits"]["train"][0])
    (prepared / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="共享源视频"):
        build_caches(prepared, "unused-model", tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
