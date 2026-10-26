"""随机微型原生Qwen的生成/缓存及评测产物检查，不下载权重、不训练。"""

import json
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest
import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

from mem_lite import evaluate_m1, evaluate_m2
from mem_lite.datasets.m1_dataset import M1Collator
from mem_lite.evaluation.runtime import parser_for
from mem_lite.models.m1_segmenter import M1Segmenter
from mem_lite.models.state_decoder import StateTextDecoder
from mem_lite.tests.test_dataset_build import configs as configs, dataset as dataset, prepared as prepared
from mem_lite.tests.test_m1_data import video_path as video_path
from mem_lite.tests.test_stage2_training import memory_batch, model as model
from mem_lite.tests.test_state_decoder import checkpoint as checkpoint
from mem_lite.train_stage2 import trainable_state


@pytest.fixture(scope="module")
def processor(checkpoint):
    path, original = checkpoint
    vision = original.config.vision_config
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            min_pixels=1024, max_pixels=2048, patch_size=vision.patch_size,
            temporal_patch_size=vision.temporal_patch_size, merge_size=vision.spatial_merge_size),
        tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(), chat_template=tokenizer.chat_template,
    )
    processor.save_pretrained(path)
    return AutoProcessor.from_pretrained(path, local_files_only=True)


def test_native_multimodal_generation_uses_only_prompt_and_resets_cache(checkpoint, processor):
    model = M1Segmenter(checkpoint[0], dtype=torch.float32, lora_rank=2, lora_alpha=4,
                        lora_dropout=0, gradient_checkpointing=False).eval()
    samples = [{"prompt": "判断切分", "images": [Image.new("RGB", (64, 48), "red")]},
               {"prompt": "判断这段视频", "images": [Image.new("RGB", (64, 48), "blue")] * 2}]
    collator = M1Collator(processor)
    batch = collator.build_prompt(samples)
    contaminated = collator.build_prompt([{**s, "response": "不能看到的真值"} for s in samples])
    assert torch.equal(batch["input_ids"], contaminated["input_ids"])
    assert "labels" not in batch
    first = model.generate(batch, max_new_tokens=3)
    assert first.shape == (2, 3)
    # 不同图像数量和左padding不会使窗口相互继承缓存；批量与单独生成一致。
    for index in range(2):
        single = model.generate(collator.build_prompt(samples[index:index + 1]), max_new_tokens=3)
        assert torch.equal(first[index], single[0])
    assert torch.equal(first, model.generate(batch, max_new_tokens=3))
    with pytest.raises(ValueError, match="不能输入"):
        model.generate({**batch, "labels": batch["input_ids"]})


def test_native_history_generation_matches_first_logits_and_keeps_new_tokens_only(checkpoint):
    decoder = StateTextDecoder(checkpoint[0], history_dim=16, dtype=torch.float32).eval()
    history = torch.randn(2, 128, 16)
    with torch.no_grad():
        prompt = decoder.build_prompt(history)
        logits = decoder.language_model(**prompt, use_cache=False, logits_to_keep=1).logits[:, -1]
    result = decoder.generate(history, max_new_tokens=3)
    assert result.shape == (2, 3)  # inputs_embeds没有被当成生成结果返回。
    assert torch.equal(result[:, 0], logits.argmax(-1))
    for index in range(2):
        assert torch.equal(result[index], decoder.generate(history[index:index + 1], max_new_tokens=3)[0])
    assert torch.equal(result, decoder.generate(history, max_new_tokens=3))
    assert all(p.grad is None for p in decoder.parameters())


def test_history_encoding_needs_no_state_labels_and_resets_between_queues(model):
    model.eval()
    batch = memory_batch(model)
    batch.pop("state_descriptions")
    with torch.inference_mode():
        histories = model.encode_history(batch)
        separate = {"pixel_values": batch["pixel_values"][-4:], "image_grid_thw": batch["image_grid_thw"][-1:],
                    "event_frame_indices": [[0]], "queue_event_counts": [1]}
        independent = model.encode_history(separate)
    assert histories.shape == (3, 128, 16)
    torch.testing.assert_close(histories[-1], independent[0], rtol=1e-4, atol=1e-5)


@pytest.fixture
def saved_checkpoint(tmp_path, checkpoint, model, processor):
    config = {
        "model_path": str(checkpoint[0]), "m1": {"lora_rank": 2, "lora_alpha": 4, "lora_dropout": 0},
        "event_encoder": {"hidden_dim": 16, "num_heads": 2}, "m2_compressor": {"hidden_dim": 16, "num_heads": 2},
        "vision_batch_size": 2, "max_pixels": 2048, "max_sequence_length": 8192,
        "max_text_sequence_length": 2048,
    }
    path = tmp_path / "stage2.pt"
    torch.save({"format": "mem-lite-stage2-v1", "model": trainable_state(model), "config": config}, path)
    return path


def args_for(checkpoint, prepared, output):
    return SimpleNamespace(checkpoint=checkpoint, data_manifest=prepared, output_dir=output,
                           model_path=None, split="validation", device="cpu", batch_size=2,
                           max_new_tokens=3, limit=None, prefix_batch_size=3)


@pytest.mark.parametrize("stage", ["m1", "m2"])
def test_evaluation_entries_generate_and_save_only_validation_samples(stage, saved_checkpoint, prepared, tmp_path):
    output = tmp_path / stage
    args = args_for(saved_checkpoint, prepared, output)
    if stage == "m1":
        args.limit = 3
    # 有意使用真实微型模型生成随机输出，以验证两个正式入口的连通性，而非伪造准确率。
    with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("评测不允许优化器更新")):
        metrics = (evaluate_m1 if stage == "m1" else evaluate_m2).evaluate(args)
    manifest = json.loads((output / "manifest.json").read_text())
    source = json.loads(prepared.read_text())
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    allowed = {video["video_id"] for video in source["videos"] if video["split"] == "validation"}
    forbidden = {video["video_id"] for video in source["videos"] if video["split"] == "train"}
    assert rows and {row["video_id"] for row in rows} <= allowed
    assert not {row["video_id"] for row in rows} & forbidden
    assert manifest["status"] == "complete" and manifest["split"] == "validation"
    assert len(manifest["checkpoint_sha256"]) == 64
    assert json.loads((output / "metrics.json").read_text()) == metrics
    assert all(len(row["generated_token_ids"]) <= 3 and "target" in row and "scores" in row for row in rows)
    if stage == "m1":
        assert len(rows) == metrics["overall"]["samples"] == 3 and manifest["partial"]
    else:
        assert len(rows) == manifest["evaluated_prefixes"] == metrics["overall"]["prefixes"] == 8
        assert len({(row["video_id"], row["sample_id"], row["prefix_length"]) for row in rows}) == len(rows)
        assert not manifest["partial"] and "M1 predicted boundaries not used" in manifest["history_source"]


def test_m1_accepts_stage1_checkpoint_and_records_failures(saved_checkpoint, prepared, tmp_path):
    ckpt = torch.load(saved_checkpoint, weights_only=True)
    torch.save({"format": "mem-lite-m1-lora-v1", "lora": ckpt["model"]["m1_lora"],
                "config": {**ckpt["config"], **ckpt["config"]["m1"]}}, saved_checkpoint)
    args = args_for(saved_checkpoint, prepared, tmp_path / "stage1-evaluation")
    args.limit = 1
    assert evaluate_m1.evaluate(args)["overall"]["samples"] == 1
    args.output_dir = tmp_path / "failed"
    with patch.object(M1Segmenter, "generate", side_effect=RuntimeError("generation failed")):
        with pytest.raises(RuntimeError, match="generation failed"):
            evaluate_m1.evaluate(args)
    metadata = json.loads((args.output_dir / "manifest.json").read_text())
    assert metadata["status"] == "failed" and "generation failed" in metadata["error"]
    assert not (args.output_dir / "metrics.json").exists()


def test_eval_default_split_is_validation():
    args = parser_for("evaluation", 96).parse_args([
        "--checkpoint", "model.pt", "--data-manifest", "manifest.json", "--output-dir", "results"])
    assert args.split == "validation"
