"""Check the transplanted heads and native-video readout on a tiny real Qwen3.5."""

import tempfile
import unittest

import numpy as np
import torch

from simplememvla.data.reward import SimpleMemRewardCollator, build_reward_messages
from simplememvla.model import SimpleMemForRewardPrediction, SimpleMemRewardConfig
from simplememvla.model.modeling_simplemem_reward import video_groups
from tests.test_progress import tiny_backbone


class SimpleMemRewardTests(unittest.TestCase):
    def make_model(self, loss_type="discrete"):
        backbone = tiny_backbone().cpu().float()
        backbone.config.video_token_id = 293
        return SimpleMemForRewardPrediction(
            SimpleMemRewardConfig(progress_loss_type=loss_type, preference_token_id=294, head_dropout=0), backbone
        )

    def batch(self, pair=False):
        # Four patch tokens per temporal group; timestamps / delimiters separate groups.
        ids = [3, 4, 290, 293, 293, 293, 293, 291, 5, 290, 293, 293, 293, 293, 291]
        if pair:
            ids += [6, 290, 293, 293, 293, 293, 291, 294]
        ids = torch.tensor([ids])
        grids = torch.tensor([[2, 4, 4]] + ([[1, 4, 4]] if pair else []))
        return dict(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            video_grid_thw=grids,
            pixel_values_videos=torch.randn(int(grids.prod(-1).sum()), 3 * 2 * 16 * 16),
            mm_token_type_ids=(ids == 293).long() * 2,
        )

    def test_grouping_skips_timestamps_padding_and_images(self):
        ids = torch.tensor([[2, 293, 293, 8, 293, 293, 292, 293], [2, 293, 293, 0, 0, 0, 0, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0, 0, 0]])
        groups, owners = video_groups(ids, mask, torch.tensor([[2, 2, 4], [1, 2, 4]]), 293, 2)
        self.assertEqual(groups[0].tolist(), [-1, 0, 0, -1, 1, 1, -1, -1])
        self.assertEqual(owners.tolist(), [[0, 0], [0, -1]])
        with self.assertRaises(ValueError):
            video_groups(ids, mask, torch.tensor([[3, 2, 4]]), 293, 2)

    def test_all_heads_and_real_video_backward(self):
        model = self.make_model().train()
        self.assertFalse(hasattr(model, "action_head"))
        out = model(
            **self.batch(pair=True),
            progress_targets=torch.tensor([[0.0, 0.4, 0.8]]),
            success_targets=torch.tensor([[0.0, 0.0, 1.0]]),
            preference_targets=torch.tensor([1.0]),
        )
        self.assertEqual(out.progress_logits.shape, (1, 3, 10))
        self.assertEqual(out.video_indices.tolist(), [[0, 0, 1]])
        self.assertTrue(torch.isfinite(out.loss))
        out.loss.backward()
        for head in (model.progress_head, model.success_head, model.preference_head):
            self.assertGreater(head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone.model.visual.patch_embed.proj.weight.grad.abs().sum().item(), 0)

    def test_future_video_does_not_change_past_progress(self):
        model = self.make_model().eval()
        batch = self.batch()
        with torch.no_grad():
            first = model(**batch)
            batch["pixel_values_videos"][16:] += 5
            second = model(**batch)
        torch.testing.assert_close(first.progress[:, :1], second.progress[:, :1])
        self.assertIsNone(first.preference_logits)

    def test_regression_masking_and_missing_preference(self):
        for kind in ("l1", "l2"):
            model = self.make_model(kind)
            out = model(**self.batch(), progress_targets=torch.tensor([[0.2, -100.0]]))
            self.assertIsNone(out.progress_logits)
            expected = (out.progress[0, 0] - 0.2).abs()
            if kind == "l2":
                expected = expected.square()
            torch.testing.assert_close(out.loss, expected)
            with self.assertRaises(ValueError):
                model(**self.batch(), preference_targets=torch.tensor([1.0]))

    def test_save_load(self):
        model = self.make_model().eval()
        batch = self.batch(pair=True)
        with torch.no_grad(), tempfile.TemporaryDirectory() as folder:
            before = model(**batch)
            model.save_pretrained(folder)
            restored = SimpleMemForRewardPrediction.from_pretrained(folder).eval()
            after = restored(**batch)
        torch.testing.assert_close(before.progress, after.progress)
        torch.testing.assert_close(before.preference_logits, after.preference_logits)

    def test_variable_length_batch_matches_individual_predictions(self):
        model = self.make_model().eval()
        long = self.batch(pair=True)
        short = self.batch()
        # Compare progress-only samples of different lengths, with right padding.
        long["input_ids"][long["input_ids"] == 294] = 7
        delta = long["input_ids"].shape[1] - short["input_ids"].shape[1]
        batch = {
            key: torch.cat([torch.nn.functional.pad(short[key], (0, delta)), long[key]])
            for key in ("input_ids", "attention_mask", "mm_token_type_ids")
        }
        for key in ("video_grid_thw", "pixel_values_videos"):
            batch[key] = torch.cat([short[key], long[key]])
        with torch.no_grad():
            out = model(**batch)
            one = model(**short)
        torch.testing.assert_close(out.progress[0, :2], one.progress[0], atol=1e-6, rtol=1e-5)
        self.assertEqual(out.progress_mask.tolist(), [[True, True, False], [True, True, True]])
        self.assertEqual(out.progress[0, 2].item(), 0)

    def test_collator_labels_and_padding(self):
        # Exercise the full collator around deterministic processor outputs.
        from types import SimpleNamespace

        batches = [self.batch(), self.batch(pair=True)]

        class Processor:
            tokenizer = SimpleNamespace(pad_token_id=0)

            def apply_chat_template(self, messages, **kwargs):
                return str(messages)

            def __call__(self, **kwargs):
                return batches.pop(0)

        features = [
            dict(instruction="press", videos=[None], video_metadata=[{}], progress_targets=[0.0, 0.5]),
            dict(instruction="press", videos=[None, None], video_metadata=[{}, {}], progress_targets=[0.0, 0.5, 1.0]),
        ]
        batch = SimpleMemRewardCollator(Processor())(features)
        self.assertEqual(batch["progress_targets"].tolist(), [[0.0, 0.5, -100.0], [0.0, 0.5, 1.0]])
        self.assertEqual(batch["video_grid_thw"].shape[0], 3)
        batches.append(self.batch())
        features[0]["progress_targets"] = [0.0, 0.2, 0.4, 0.6]
        with self.assertRaisesRegex(ValueError, "temporal-patch labels"):
            SimpleMemRewardCollator(Processor())(features[:1])

    def test_preference_token_cannot_precede_video(self):
        model = self.make_model()
        batch = self.batch(pair=True)
        batch["input_ids"][0, 0] = 294
        batch["input_ids"][0, -1] = 7
        with self.assertRaisesRegex(ValueError, "after all visual"):
            model(**batch)

    def test_task_precedes_video_and_pair_token_follows_both(self):
        content = build_reward_messages("press twice", paired=True, current_cameras=1)[0]["content"]
        self.assertIn("press twice", content[0]["text"])
        self.assertEqual(sum(item["type"] == "video" for item in content), 2)
        self.assertTrue(content[-1]["text"].endswith("<|pref_token|>"))


class StreamingTests(unittest.TestCase):
    make_model = SimpleMemRewardTests.make_model
    batch = SimpleMemRewardTests.batch

    def test_append_replay_replace_and_slide(self):
        from simplememvla.model.streaming import SimpleMemRewardStream

        model = self.make_model().eval()
        stream = SimpleMemRewardStream(model)
        full = self.batch()
        first = {k: v.clone() for k, v in full.items()}
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            first[key] = first[key][:, :8]
        first["video_grid_thw"][0, 0] = 1
        first["pixel_values_videos"] = first["pixel_values_videos"][:16]
        for batch, keys in [(first, [(0, 1)]), (full, [(0, 1), (2, 3)]), (full, [(0, 1), (2, 3)])]:
            with torch.inference_mode():
                expected = model(**batch)
                actual = stream.step(batch, keys)
            torch.testing.assert_close(actual.progress_logits, expected.progress_logits, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(actual.success_logits, expected.success_logits, atol=2e-5, rtol=2e-5)
        self.assertEqual(stream.stats["vision_patches"], 0)
        self.assertEqual(stream.stats["language_tokens"], 0)
        full["pixel_values_videos"][16:] += 2
        with torch.inference_mode():
            expected = model(**full)
            actual = stream.step(full, [(0, 1), (2, 4)])
        self.assertTrue(stream.stats["rebuilt"])
        self.assertEqual(stream.stats["vision_patches"], 1)
        torch.testing.assert_close(actual.progress_logits, expected.progress_logits, atol=2e-5, rtol=2e-5)
        with torch.inference_mode():
            first["pixel_values_videos"] = full["pixel_values_videos"][16:]
            expected = model(**first)
            actual = stream.step(first, [(2, 4)])
        self.assertTrue(stream.stats["rebuilt"])
        self.assertEqual(len(stream.features), 1)
        torch.testing.assert_close(actual.progress_logits, expected.progress_logits, atol=2e-5, rtol=2e-5)

    def test_task_change_and_episode_reset(self):
        from simplememvla.model.streaming import SimpleMemRewardStream

        model = self.make_model().eval()
        stream = SimpleMemRewardStream(model)
        batch = self.batch()
        keys = [(0, 1), (2, 3)]
        stream.step(batch, keys)
        batch["input_ids"][0, 0] += 1
        with torch.inference_mode():
            expected = model(**batch)
            actual = stream.step(batch, keys)
        self.assertTrue(stream.stats["rebuilt"])
        self.assertEqual(stream.stats["vision_patches"], 0)
        torch.testing.assert_close(actual.success_logits, expected.success_logits, atol=2e-5, rtol=2e-5)
        stream.reset()
        self.assertIsNone(stream.cache)
        self.assertFalse(stream.features)
        model.train()
        with self.assertRaisesRegex(ValueError, "model.eval"):
            stream.step(batch, keys)


class CPUPreprocessorTests(unittest.TestCase):
    def test_native_inputs_predictions_and_reset(self):
        from tokenizers.pre_tokenizers import ByteLevel
        from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLProcessor
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        from transformers.video_utils import VideoMetadata

        from simplememvla.data.streaming import SimpleMemRewardPreprocessor
        from simplememvla.model import SimpleMemRewardStream

        vocab = {char: i for i, char in enumerate(sorted(ByteLevel.alphabet()))}
        vocab.update({f"unused{i}": i for i in range(256, 290)})
        special = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>", "<|im_end|>"]
        vocab.update({token: 290 + i for i, token in enumerate(special)})
        tokenizer = Qwen2TokenizerFast(
            vocab=vocab, merges=[], additional_special_tokens=special
        )
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessorFast(),
            video_processor=Qwen3VLVideoProcessor(),
            tokenizer=tokenizer,
            chat_template=("{{messages[0]['content'][0]['text']}} A:"
                "<|vision_start|><|video_pad|><|vision_end|><|im_end|>"),
        )
        cache = SimpleMemRewardPreprocessor(processor, "Press twice", 2.0)
        model = SimpleMemRewardTests().make_model().eval()
        model.config.preference_token_id = 299
        stream = SimpleMemRewardStream(model)
        frames = np.random.default_rng(7).integers(0, 256, (10, 32, 96, 3), dtype=np.uint8)
        previous = 0
        for end in (2, 6, 8, 10):
            actual = cache.append(frames[previous:end], range(previous, end))
            processor.video_processor.size = {"shortest_edge": end * 32 * 96, "longest_edge": end * 32 * 96}
            expected = SimpleMemRewardCollator(processor)([{
                "instruction": "Press twice", "videos": [frames[:end]],
                "video_metadata": [VideoMetadata(total_num_frames=end, fps=2.0, frames_indices=list(range(end)))],
            }])
            for key in expected:
                self.assertTrue(torch.equal(actual[key], expected[key]), key)
            with torch.inference_mode():
                reference = model(**expected)
                prediction = stream.step(actual, cache.patch_keys)
            torch.testing.assert_close(prediction.progress_logits, reference.progress_logits, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(prediction.success_logits, reference.success_logits, atol=2e-5, rtol=2e-5)
            self.assertEqual(cache.stats["processed_frames"], end - previous)
            previous = end
        with self.assertRaisesRegex(ValueError, "increase"):
            cache.append(frames[:2], [0, 1])
        with self.assertRaisesRegex(ValueError, "complete"):
            cache.append(frames[:1], [10])
        with self.assertRaisesRegex(ValueError, "fixed resolution"):
            cache.append(frames[:2, :, :32], [10, 11])
        cache.reset()
        stream.reset()
        replay = cache.append(frames[:2], [100, 101])
        self.assertEqual(cache.patch_keys, [(100, 101)])
        self.assertEqual(replay["video_grid_thw"][0, 0], 1)


if __name__ == "__main__":
    unittest.main()
