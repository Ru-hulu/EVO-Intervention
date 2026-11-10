"""Global-depth semantics, fixed budgets, source identity and mode safety."""

from copy import deepcopy
import json
import math
import random
import shutil
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from mem_lite.memory.clip import ClipBoundaryEncoder
from mem_lite.memory.manager import MemoryManager
from mem_lite.memory.splitting import DEPTH_SPEC, global_depth_scores, override_memory_config, validate_history_source
from mem_lite.memory.stream import StreamMemory
from mem_lite.train_stage3 import configure_history


def test_cli_can_switch_to_original_without_loading_clip():
    config = override_memory_config({"event_split_method": "max_depth", "clip_model_path": "/missing"},
                                    "min_similarity")
    assert "clip_model_path" not in config
    stream = StreamMemory(SimpleNamespace(hidden_size=8), None, config)
    assert stream.boundary_encoder is None
    assert stream.memory.event_split_method == "min_similarity"


def test_depth_matches_exclusive_global_formula_including_edges():
    random.seed(42)
    for count in range(1, 100):
        values = [random.uniform(-1, 1) for _ in range(count)]
        reference = [max(values[:i], default=v) + max(values[i + 1:], default=v) - 2 * v
                     for i, v in enumerate(values)]
        assert global_depth_scores(values) == pytest.approx(reference, abs=0, rel=0)
    assert global_depth_scores([]) == []
    assert global_depth_scores([0.7]) == [0]
    assert global_depth_scores([0.7] * 4) == [0] * 4
    # The distant peak must remain visible across an intervening local valley.
    assert global_depth_scores([1, 0.4, 0.5, 0.3, 0.8])[3] == pytest.approx(1.2)
    with pytest.raises(ValueError, match="有限"):
        global_depth_scores([1, float("nan")])


def test_depth_uses_clip_and_preserves_trigger_candidate_range_and_tokens():
    managers = [MemoryManager(8, st_memory_windows=(1, 1), st_memory_tokens=(4, 2),
                              event_split_window=4, long_memory_tokens_per_frame=1,
                              long_memory_tokens_quota=10, event_split_method=method)
                for method in ("min_similarity", "max_depth")]
    angle = 0
    for index, cosine in enumerate([1, 0.1, 0.9, 0.2, 0.9, 0.5]):
        angle += math.acos(cosine)
        descriptor = torch.tensor([math.cos(angle), math.sin(angle)])
        for manager in managers:
            manager.update(torch.ones(8, 8), descriptor if manager.event_split_method == "max_depth" else None)
            if index < 5:
                assert manager.frames_in_long == 0
    assert managers[0].frames_in_long == 1
    assert managers[1].frames_in_long == 3  # max depth at gap 3, not minimum similarity at gap 1
    for manager in managers:
        assert manager.get_memory_tokens().shape[1] == 8  # memory is Qwen, not 2D CLIP
        assert len(manager.get_memory_tokens()) <= manager.token_budget


@pytest.mark.parametrize("window", [1, 2, 4])
def test_static_frames_tie_earliest_and_bounded_state(window):
    manager = MemoryManager(8, st_memory_windows=(1, 2), st_memory_tokens=(4, 2),
                            event_split_window=window, long_memory_tokens_per_frame=1,
                            long_memory_tokens_quota=4, event_split_method="max_depth")
    descriptor = torch.ones(6)
    for index in range(150):
        previous = manager.frames_in_long
        manager.update(torch.ones(8, 8), descriptor)
        assert manager.frames_in_long - previous <= 1
        assert len(manager.get_memory_tokens()) <= manager.token_budget
        assert len(manager.frame_similarities) == manager.frames_seen - manager.frames_in_long - 1
        assert len(manager.frame_similarities) < 1 + 2 + window
    manager.reset()
    assert manager.last_frame is None and not manager.frame_similarities
    with pytest.raises(ValueError, match="CLIP"):
        manager.update(torch.ones(8, 8))


def test_clip_is_frozen_cls_and_autocast_independent(clip_checkpoint):
    encoder = ClipBoundaryEncoder(clip_checkpoint, "cpu")
    image = Image.new("RGB", (48, 32), "red")
    pixels = encoder.processor(images=[image], return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        raw = encoder.model(pixel_values=pixels)
        expected = encoder.model.vision_model.post_layernorm(raw.last_hidden_state[:, 0])[0]
    actual = encoder(image)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        other = encoder(image)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(other, actual, atol=0, rtol=0)
    assert not actual.requires_grad
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_clip_provenance_path_independent_but_processor_and_weights_sensitive(clip_checkpoint, tmp_path):
    encoder = ClipBoundaryEncoder(clip_checkpoint, "cpu")
    copy = tmp_path / "moved-clip"
    shutil.copytree(clip_checkpoint, copy)
    assert ClipBoundaryEncoder(copy, "cpu").provenance == encoder.provenance
    config = json.loads((copy / "preprocessor_config.json").read_text())
    config["image_mean"][0] += 0.1
    (copy / "preprocessor_config.json").write_text(json.dumps(config))
    altered = ClipBoundaryEncoder(copy, "cpu")
    assert altered.provenance["processor_sha256"] != encoder.provenance["processor_sha256"]
    with torch.no_grad():
        next(altered.model.parameters()).add_(1)
    altered.model.save_pretrained(copy)
    assert ClipBoundaryEncoder(copy, "cpu").provenance["weights_sha256"] != encoder.provenance["weights_sha256"]


def test_missing_clip_weights_are_rejected_instead_of_randomly_initialized(clip_checkpoint, tmp_path):
    from safetensors.torch import load_file, save_file

    copy = tmp_path / "incomplete-clip"
    shutil.copytree(clip_checkpoint, copy)
    weights = load_file(copy / "model.safetensors")
    del weights["vision_model.embeddings.class_embedding"]
    save_file(weights, copy / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="CLIP 视觉权重不完整"):
        ClipBoundaryEncoder(copy, "cpu")


def test_metadata_rejects_missing_provenance_and_wrong_training_request(clip_checkpoint):
    old = {"memory_config": {}}
    assert validate_history_source(old) == "min_similarity"
    source = {"memory_config": {"event_split_method": "max_depth"}}
    with pytest.raises(ValueError, match="元数据"):
        validate_history_source(source)
    source["event_split"] = {**DEPTH_SPEC, "clip": ClipBoundaryEncoder(clip_checkpoint, "cpu").provenance}
    assert validate_history_source(source) == "max_depth"
    dataset = SimpleNamespace(history_source=source, history_dim=8, history_capacity=16)
    with pytest.raises(ValueError, match="切分方式"):
        configure_history({"model": {}, "event_split_method": "min_similarity"}, dataset)
    with pytest.raises(ValueError, match="history_source"):
        configure_history({"model": {}, "history_source": old}, dataset)
    config = {"model": {}}
    configure_history(config, dataset)
    assert config["event_split_method"] == "max_depth"
    assert config["history_source"] == source
    broken = deepcopy(source)
    broken["event_split"]["depth"] = "local_peaks"
    with pytest.raises(ValueError, match="元数据"):
        validate_history_source(broken)


def test_stream_rejects_wrong_clip_source_before_observations(clip_checkpoint):
    encoder = torch.nn.Linear(3, 3)
    encoder.hidden_size = 3
    config = {"event_split_method": "max_depth", "clip_model_path": str(clip_checkpoint)}
    stream = StreamMemory(encoder, None, config)
    source = {"memory_config": config, "max_pixels": 65536, "min_pixels": 4096}
    with pytest.raises(ValueError, match="元数据"):
        stream.bind_history_source(source, require_provenance=True)
    source = stream.bind_history_source(source)
    assert stream.bind_history_source(source, require_provenance=True) == source
    source["event_split"]["clip"]["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="CLIP 权重"):
        stream.bind_history_source(source, require_provenance=True)
