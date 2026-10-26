"""批量数据构建、视频级隔离、可迁移路径及失败续建；不加载模型或启动训练。"""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from mem_lite.datasets import prepare_dataset as builder
from mem_lite.datasets.dataset_manifest import create_manifest, load_sample_files
from mem_lite.datasets.m1_dataset import M1Dataset
from mem_lite.datasets.m2_dataset import M2Dataset, Stage2Dataset
from mem_lite.datasets.prepare_m1 import DEFAULT_CONFIG as M1_CONFIG
from mem_lite.datasets.prepare_m2 import DEFAULT_CONFIG as M2_CONFIG
from mem_lite.tests.test_m1_data import video_path as video_path


@pytest.fixture
def configs():
    m1, m2 = json.loads(M1_CONFIG.read_text()), json.loads(M2_CONFIG.read_text())
    m1["samples_per_case"] = {case: 2 for case in m1["samples_per_case"]}
    m2["draws_per_video"] = 2
    return m1, m2


@pytest.fixture
def dataset(tmp_path, video_path):
    root = tmp_path / "source"
    for task in ("task-a", "task-b"):
        directory = root / task / "annotations"
        directory.mkdir(parents=True)
        for index in range(2):
            name = f"episode_{index}.mp4"  # 跨任务同名视频不能碰撞输出路径。
            shutil.copyfile(video_path, directory.parent / name)
            label = {
                "video_file": name, "task_prompt": "点击两次", "fps": 25, "total_frames": 50,
                "frame_index_base": 0, "frame_range_convention": "[start_frame, end_frame)",
                "stages": [
                    {"stage_id": 0, "start_frame": 0, "end_frame": 15, "state_description": "尚未点击"},
                    {"stage_id": 1, "start_frame": 15, "end_frame": 35, "state_description": "已点击一次"},
                    {"stage_id": 2, "start_frame": 35, "end_frame": 50, "state_description": "已点击两次"},
                ],
            }
            (directory / f"episode_{index}.annotations.json").write_text(json.dumps(label), encoding="utf-8")
        (directory.parent / "unannotated.mp4").write_bytes(b"must not decode")
        wrist = directory.parent / "cam_left_wrist"
        wrist.mkdir()
        (wrist / "episode_0.mp4").write_bytes(b"must not decode")
    return root


@pytest.fixture
def prepared(tmp_path, dataset, configs):
    output = tmp_path / "prepared"
    builder.prepare_dataset(dataset, output, *configs)
    return output / "manifest.json"


def test_split_is_video_level_reproducible_and_independent_of_root(dataset, configs, tmp_path):
    first = create_manifest(dataset, *configs)
    copy = tmp_path / "relocated-source"
    shutil.copytree(dataset, copy)
    second = create_manifest(copy, *configs)
    assert first["videos"] == second["videos"] and first["summary"] == second["summary"]
    assert len(first["videos"]) == 4
    for split in ("train", "validation"):
        assert first["summary"][split]["by_task"] == {"task-a": 1, "task-b": 1}
    train = {v["video_id"] for v in first["videos"] if v["split"] == "train"}
    validation = {v["video_id"] for v in first["videos"] if v["split"] == "validation"}
    assert train.isdisjoint(validation) and len(train | validation) == 4
    assert len({v["seeds"]["m1"] for v in first["videos"]}) == 4
    assert len({v["seeds"]["m2"] for v in first["videos"]}) == 4
    assert all(not Path(path).is_absolute() for v in first["videos"] for path in v["samples"].values())


def test_singletons_zero_fraction_and_duplicate_sources(dataset, configs):
    (dataset / "task-b/annotations/episode_1.annotations.json").unlink()
    manifest = create_manifest(dataset, *configs)
    assert manifest["summary"]["train"]["by_task"]["task-b"] == 1
    assert "task-b" not in manifest["summary"]["validation"]["by_task"]
    all_train = create_manifest(dataset, *configs, validation_fraction=0)
    assert all(v["split"] == "train" for v in all_train["videos"])
    for fraction in (-0.1, 1, float("nan")):
        with pytest.raises(ValueError, match="validation_fraction"):
            create_manifest(dataset, *configs, validation_fraction=fraction)
    directory = dataset / "task-a/annotations"
    shutil.copyfile(directory / "episode_0.annotations.json", directory / "duplicate.annotations.json")
    with pytest.raises(ValueError, match="重复|多份"):
        create_manifest(dataset, *configs)


def test_real_video_build_load_and_move_keep_both_stages_separate(prepared, tmp_path):
    manifest = json.loads(prepared.read_text())
    assert manifest["status"] == "complete"
    for split in ("train", "validation"):
        m1_files, m2_files = (load_sample_files(prepared, stage, split) for stage in ("m1", "m2"))
        assert len(m1_files) == len(m2_files) == 2
        m1, m2 = M1Dataset(m1_files), M2Dataset(m2_files)
        assert len(m1) == manifest["summary"][split]["m1_samples"]
        assert len(m2) == manifest["summary"][split]["m2_queues"] == 4
        assert manifest["summary"][split]["m2_prefixes"] == 8
        assert m1[0]["images"] and m2[0]["images"]
        assert m2[0]["state_descriptions"] == ["已点击一次", "已点击两次"]
        paired = Stage2Dataset(m2_files, m1_files)
        assert paired[0]["memory"]["state_descriptions"] == ["已点击一次", "已点击两次"]
        expected = {v["video_id"] for v in manifest["videos"] if v["split"] == split}
        for stage, files in (("m1", m1_files), ("m2", m2_files)):
            actual = {str(Path(path).parent.relative_to(prepared.parent / stage)) for path in files}
            assert actual == expected
    relocated = tmp_path / "moved"
    shutil.move(prepared.parent, relocated)
    for stage, loader in (("m1", M1Dataset), ("m2", M2Dataset)):
        paths = load_sample_files(relocated / "manifest.json", stage)
        assert all(str(relocated) in path for path in paths)
        assert loader(paths)[0]["images"]


def test_plan_does_not_decode_and_changed_labels_cannot_resume(dataset, configs, tmp_path, monkeypatch):
    def should_not_run(*args, **kwargs):
        raise AssertionError("计划阶段不能解码视频")

    monkeypatch.setattr(builder, "prepare_m1", should_not_run)
    output = tmp_path / "plan"
    manifest = builder.prepare_dataset(dataset, output, *configs, plan_only=True)
    assert manifest["status"] == "planned"
    assert [p.name for p in output.iterdir()] == ["manifest.json"]
    with pytest.raises(ValueError, match="尚未完整"):
        load_sample_files(output / "manifest.json", "m1")
    annotation = dataset / manifest["videos"][0]["annotation"]
    label = json.loads(annotation.read_text())
    label["task_prompt"] = "changed"
    annotation.write_text(json.dumps(label))
    with pytest.raises(ValueError, match="标注已改变"):
        builder.build_dataset(output / "manifest.json")


def test_failure_can_resume_without_resampling_or_overwriting_completed_data(dataset, configs, tmp_path, monkeypatch):
    original_m1, original_m2 = builder.prepare_m1, builder.prepare_m2
    m1_calls, fail = [], [True]

    def m1(*args, **kwargs):
        m1_calls.append(str(args[0]))
        return original_m1(*args, **kwargs)

    def m2(annotation, output, config, video):
        if fail[0]:
            fail[0] = False
            output.mkdir(parents=True)
            (output / "incomplete-marker").write_text("interrupted")
            raise RuntimeError("模拟一次中断")
        return original_m2(annotation, output, config, video)

    monkeypatch.setattr(builder, "prepare_m1", m1)
    monkeypatch.setattr(builder, "prepare_m2", m2)
    output = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="中断"):
        builder.prepare_dataset(dataset, output, *configs)
    path = output / "manifest.json"
    failed = json.loads(path.read_text())
    assert failed["status"] == "failed"
    first_samples = output / failed["videos"][0]["samples"]["m1"]
    original_bytes, modified_time = first_samples.read_bytes(), first_samples.stat().st_mtime_ns
    with pytest.raises(ValueError, match="尚未完整"):
        load_sample_files(path, "m1")
    completed = builder.build_dataset(path)
    assert completed["status"] == "complete" and len(m1_calls) == 4
    assert first_samples.read_bytes() == original_bytes and first_samples.stat().st_mtime_ns == modified_time
    assert [(v["split"], v["seeds"]) for v in failed["videos"]] == [(v["split"], v["seeds"]) for v in completed["videos"]]
    assert not list(output.rglob("*.partial")) and not list(output.rglob("incomplete-marker"))
    with pytest.raises(FileExistsError):
        builder.prepare_dataset(dataset, output, *configs)


def test_training_entry_configs_use_only_train_and_share_video_membership(prepared, tmp_path):
    from mem_lite.train_m1 import DEFAULT_CONFIG as M1_TRAIN, read_config as read_m1
    from mem_lite.train_stage2 import DEFAULT_CONFIG as STAGE2_TRAIN, read_config as read_stage2

    common = {"model_path": str(tmp_path / "model"), "output_dir": None, "epochs": None,
              "max_steps": None, "data_manifest": str(prepared)}
    first_args = SimpleNamespace(**common, config=str(M1_TRAIN), sample_files=None)
    second_args = SimpleNamespace(**common, config=str(STAGE2_TRAIN), m1_checkpoint="/m1.pt",
                                  m1_sample_files=None, m2_sample_files=None)
    first, second = read_m1(first_args, None), read_stage2(second_args)
    assert first["sample_files"] == second["m1_sample_files"] == load_sample_files(prepared, "m1", "train")
    assert second["m2_sample_files"] == load_sample_files(prepared, "m2", "train")
    assert set(first["sample_files"]).isdisjoint(load_sample_files(prepared, "m1", "validation"))
    assert set(second["m2_sample_files"]).isdisjoint(load_sample_files(prepared, "m2", "validation"))
    first_args.sample_files = ["/manual"]
    with pytest.raises(ValueError, match="同时指定"):
        read_m1(first_args, None)
    second_args.m2_sample_files = ["/manual"]
    with pytest.raises(ValueError, match="同时指定"):
        read_stage2(second_args)
