"""缓存构造、在线预测、真实微型训练步骤与检查点加载的闭环检查。"""

import json
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from mem_lite.datasets.m3_dataset import M3Dataset, Stage3Collator
from mem_lite.datasets.prepare_stream import build_episode
from mem_lite.evaluate import evaluate
from mem_lite.memory.stream import StreamMemory
from mem_lite.memory.vision import QwenFrameEncoder, _vision_sha256, verify_vision_source
from mem_lite.models.m3_predictor import M3ProgressPredictor
from mem_lite.runtime import StreamingProgressPredictor, load_checkpoint
from mem_lite.tests.test_m3_predictor import collator as collator
from mem_lite.train_stage3 import CHECKPOINT_FORMAT, DEFAULT_CONFIG, configure_history, save_checkpoint


@pytest.fixture
def origin(checkpoint):
    return {"method": "streamforest-v1", "vision_sha256": _vision_sha256(checkpoint[1].model.visual.state_dict()),
            "frame_stride": 3,
            "max_pixels": 4096, "min_pixels": 1024,
            "memory_config": {"st_memory_windows": [1, 2], "st_memory_tokens": [4, 2],
                              "event_split_window": 2, "long_memory_tokens_per_frame": 1,
                              "long_memory_tokens_quota": 4}}


def make_stream(visual, processor, origin):
    return StreamMemory(QwenFrameEncoder(visual), processor.image_processor, origin["memory_config"],
                        origin["max_pixels"], origin["min_pixels"])


def observations():
    return [{"frame_index": index * 3, "pt": value,
             "images": [Image.new("RGB", (32, 32), color), Image.new("RGB", (32, 64), "green"),
                        Image.new("RGB", (64, 32), "blue")]}
            for index, (color, value) in enumerate(zip(
                ("red", "red", "blue", "green", "blue", "red", "green", "blue"),
                (0.0, 0.3, 0.1, 0.6, 0.4, 0.7, 0.9, 1.0), strict=True))]


@pytest.fixture
def pipeline(checkpoint, collator, origin, tmp_path, request):
    model = M3ProgressPredictor(checkpoint[0], dtype=getattr(request, "param", torch.float32), lora_rank=2, lora_alpha=4,
                                lora_dropout=0, gradient_checkpointing=False).eval()
    stream = make_stream(model.qwen.model.visual, collator.processor, origin)
    directory = tmp_path / "episode"
    build_episode(observations(), stream, directory, "依次完成操作", "episode", origin)
    return model, collator.processor, M3Dataset([directory]), directory


def test_stream_is_causal_and_resets_on_episode_or_task_change(checkpoint, collator, origin):
    stream = make_stream(checkpoint[1].model.visual, collator.processor, origin)
    red, blue, green = [Image.new("RGB", (32, 32), color) for color in ("red", "blue", "green")]
    assert stream.step(red, 0, "one", "task") is None
    before_blue = stream.step(blue, 3, "one", "task")
    after_blue = stream.step(red, 6, "one", "task")
    stream.reset()
    assert stream.step(red, 0, "one", "task") is None
    before_green = stream.step(green, 3, "one", "task")
    after_green = stream.step(red, 6, "one", "task")
    torch.testing.assert_close(before_blue, before_green)
    assert not torch.allclose(after_blue, after_green)
    with pytest.raises(ValueError, match="递增"):
        stream.step(red, 6, "one", "task")
    assert stream.step(red, 0, "two", "task") is None
    assert stream.step(red, 0, "two", "new task") is None


@pytest.mark.parametrize("autocast", [False, True])
def test_saved_history_matches_online_and_never_uses_labels(pipeline, origin, tmp_path, autocast):
    model, processor, dataset, _ = pipeline
    stream = make_stream(model.qwen.model.visual, processor, origin)
    for sample in dataset:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            actual = stream.step(sample["images"][0], sample["frame_index"], sample["episode_id"], sample["task_prompt"])
        if actual is None:
            assert sample["history_tokens"] is None
        else:
            torch.testing.assert_close(actual, sample["history_tokens"], atol=0, rtol=0)
            assert len(actual) <= dataset.history_capacity
    altered = [{**row, "pt": 1 - row["pt"]} for row in observations()]
    directory = tmp_path / "different-labels"
    build_episode(altered, stream, directory, "依次完成操作", "episode", origin)
    other = M3Dataset([directory])
    for index in range(1, len(dataset)):
        torch.testing.assert_close(dataset[index]["history_tokens"], other[index]["history_tokens"], atol=0, rtol=0)


@pytest.mark.parametrize("pipeline", [torch.float32, torch.bfloat16], indirect=True)
@pytest.mark.parametrize("autocast", [False, True])
def test_cached_and_online_evaluation_produce_same_predictions(pipeline, autocast):
    model, processor, dataset, directory = pipeline
    cached, cached_rows = evaluate(model, processor, dataset)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        online, online_rows = evaluate(model, processor, M3Dataset([directory], load_history=False), online=True)
    assert online["num_samples"] == cached["num_samples"] == len(dataset)
    assert online["mae_pp"] == pytest.approx(cached["mae_pp"], abs=1e-5)
    assert online["rmse_pp"] == pytest.approx(cached["rmse_pp"], abs=1e-5)
    assert [row["predicted_pt"] for row in online_rows] == pytest.approx([row["predicted_pt"] for row in cached_rows])
    assert [row["history_length"] for row in online_rows] == [row["history_length"] for row in cached_rows]
    assert online["latency_ms_p95"] > 0 and online["peak_cuda_memory_mib"] is None
    baseline, rows = evaluate(model, processor, dataset, without_history=True)
    assert baseline["mode"] == "without_history" and all(row["history_length"] == 0 for row in rows)


def test_future_frames_cannot_change_earlier_snapshots_or_predictions(pipeline, origin, tmp_path):
    model, processor, _, _ = pipeline
    stream = make_stream(model.qwen.model.visual, processor, origin)
    original = [{**row, "frame_index": index * 3} for index, row in enumerate(observations() * 4)]
    cutoff = 24  # 覆盖多次事件切分及长期合并之后的历史。
    changed = [row if index < cutoff else {**row, "images": [Image.new("RGB", (32, 32), "white")] * 3}
               for index, row in enumerate(original)]
    datasets, predictions = [], []
    for name, rows in (("original", original), ("changed-future", changed)):
        directory = tmp_path / name
        build_episode(rows, stream, directory, "task", name, origin)
        assert any(event.merges > 1 for event in stream.memory.memory_long)
        dataset = M3Dataset([directory])
        datasets.append(dataset)
        _, records = evaluate(model, processor, dataset, online=True)
        predictions.append([row["predicted_pt"] for row in records])
    # 连被替换的第一帧，也只能读取修改之前的历史；当前图像只影响当前预测及后续记忆。
    for index in range(1, cutoff + 1):
        torch.testing.assert_close(datasets[0][index]["history_tokens"], datasets[1][index]["history_tokens"],
                                   atol=0, rtol=0)
    assert predictions[0][:cutoff] == predictions[1][:cutoff]
    assert not torch.equal(datasets[0][cutoff + 1]["history_tokens"], datasets[1][cutoff + 1]["history_tokens"])


def test_loading_rejects_other_visual_weights_even_with_same_dimensions(checkpoint, origin, tmp_path):
    from safetensors.torch import load_file, save_file

    verify_vision_source(checkpoint[0], origin)
    different = tmp_path / "different-base"
    different.mkdir()
    state = load_file(checkpoint[0] / "model.safetensors")
    key = next(key for key in state if key.startswith("model.visual."))
    state[key] = state[key] + 1
    save_file(state, different / "model.safetensors")
    saved = tmp_path / "checkpoint.pt"
    torch.save({"format": CHECKPOINT_FORMAT,
                "config": {"model_path": str(checkpoint[0]), "history_source": origin}}, saved)
    with pytest.raises(ValueError, match="视觉权重"):
        load_checkpoint(saved, device="cpu", model_path=different)


def test_real_optimizer_step_checkpoint_reload_and_stream_prediction(pipeline, checkpoint, tmp_path, monkeypatch):
    model, processor, dataset, _ = pipeline
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(model_path=str(checkpoint[0]), max_pixels=4096)
    config["model"].update(lora_rank=2, lora_alpha=4, lora_dropout=0, gradient_checkpointing=False)
    configure_history(config, dataset)
    assert config["model"]["history_dim"] == model.qwen.config.vision_config.out_hidden_size
    assert config["history_capacity"] == dataset.history_capacity
    batcher = Stage3Collator(processor, dataset.history_dim, history_capacity=dataset.history_capacity)
    batch = batcher([dataset[0], dataset[5]])
    model.train()
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    before = model.history_projection.weight.detach().clone()
    result = model(batch["inputs"], batch["target_pt"])
    result["loss"].backward()
    assert model.history_projection.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.qwen.model.visual.parameters())
    optimizer.step()
    assert not torch.equal(before, model.history_projection.weight)
    model.eval()
    with torch.no_grad():
        expected = model(batch["inputs"])["pt"]
    processor.save_pretrained(checkpoint[0])
    with patch("mem_lite.train_stage3.torch.cuda.get_rng_state", return_value=torch.tensor([0], dtype=torch.uint8)):
        save_checkpoint(model, optimizer, config, tmp_path, epoch=0, next_batch=1, step=1)
    restored, restored_processor, saved_config = load_checkpoint(tmp_path / "last.pt", device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(restored(batch["inputs"])["pt"], expected)
    runner = StreamingProgressPredictor(restored, restored_processor, saved_config["history_source"])
    first = dataset[0]
    initial = runner(first["images"], first["frame_index"], first["episode_id"], first["task_prompt"])
    assert initial["history_length"] == 0
    second = dataset[1]
    actual = runner(second["images"], second["frame_index"], second["episode_id"], second["task_prompt"])
    with torch.no_grad():
        expected = restored(batcher([second])["inputs"])
    torch.testing.assert_close(actual["pt"], expected["pt"], atol=1e-6, rtol=1e-5)
    from mem_lite.evaluate import main

    output = tmp_path / "evaluation"
    monkeypatch.setattr("sys.argv", ["evaluate", "--checkpoint", str(tmp_path / "last.pt"),
                                    "--sample-files", str(pipeline[3]), "--output-dir", str(output),
                                    "--device", "cpu", "--online"])
    main()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["mode"] == "online" and metrics["num_samples"] == len(dataset)
    assert len((output / "predictions.jsonl").read_text().splitlines()) == len(dataset)


def write_video(path, color, count=10, rate=30):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(count):
            image = Image.new("RGB", (32, 32), color if index % 2 else "black")
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.mark.parametrize("rate,sampling,indices,source", [
    (25, [], [0, 5, 10], {"sample_fps": 5.0}),
    (30, ["--sample-fps", "5"], [0, 6, 12], {"sample_fps": 5.0}),
    (25, ["--frame-stride", "3"], [0, 3, 6, 9, 12], {"frame_stride": 3}),
])
def test_video_preparation_cli_with_real_tiny_visual_weights(
    checkpoint, collator, origin, tmp_path, monkeypatch, rate, sampling, indices, source,
):
    from transformers import AutoProcessor

    from mem_lite.datasets.prepare_stream import main

    videos = [tmp_path / f"{view}.mp4" for view in ("high", "left", "right")]
    for video, color in zip(videos, ("red", "green", "blue"), strict=True):
        write_video(video, color, count=13, rate=rate)
    progress = tmp_path / "progress.csv"
    progress.write_text("frame_index,delta,pt\n" + "".join(f"{i},1,{i / 12}\n" for i in range(13)))
    config = tmp_path / "memory.json"
    config.write_text(json.dumps(origin["memory_config"]))
    collator.processor.save_pretrained(checkpoint[0])
    output = tmp_path / "video-samples"
    monkeypatch.setattr("sys.argv", [
        "prepare_stream", "--videos", *map(str, videos), "--progress-csv", str(progress),
        "--task-prompt", "测试视频", "--model-path", str(checkpoint[0]), "--output-dir", str(output),
        "--memory-config", str(config), *sampling, "--device", "cpu",
        "--max-pixels", "4096", "--min-pixels", "1024",
    ])
    main()
    dataset = M3Dataset([output])
    assert [row["frame_index"] for row in dataset] == indices
    assert [row["target_pt"] for row in dataset] == pytest.approx([index / 12 for index in indices])
    assert dataset[0]["history_tokens"] is None
    assert dataset[-1]["history_tokens"].shape[1] == checkpoint[1].config.vision_config.out_hidden_size
    assert all(dataset.history_source[key] == value for key, value in source.items())
    processor = AutoProcessor.from_pretrained(checkpoint[0], max_pixels=4096, local_files_only=True)
    stream = make_stream(checkpoint[1].model.visual, processor, dataset.history_source)
    for sample in dataset:
        actual = stream.step(sample["images"][0], sample["frame_index"], sample["episode_id"], sample["task_prompt"])
        if actual is not None:
            torch.testing.assert_close(actual, sample["history_tokens"], atol=0, rtol=0)


def test_video_reader_rejects_unsynchronized_camera_streams(tmp_path):
    from mem_lite.datasets.prepare_stream import iter_observations

    videos = [tmp_path / f"{i}.mp4" for i in range(3)]
    for video, count in zip(videos, (4, 4, 3), strict=True):
        write_video(video, "red", count=count)
    with pytest.raises(ValueError):
        list(iter_observations(videos, dict.fromkeys(range(4), 0.0)))
    write_video(videos[-1], "red", count=4, rate=25)
    with pytest.raises(ValueError, match="帧率"):
        list(iter_observations(videos, dict.fromkeys(range(4), 0.0)))
