# ruff: noqa: RUF002, RUF003
"""用随机初始化的小 Qwen3.5 检查流式前向、可见性隔离、反传与数据样本。有 GPU 时在 GPU 上跑（走 fla kernel）。"""

import json
import math
import os
import random
import tempfile
import unittest
from collections import Counter
from itertools import pairwise
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from qwen_training.datasets.rmbench.labels import MAX_COUNT, episode_labels
from qwen_training.datasets.rmbench.tasks import TASKS
from qwen_training.progress import data
from qwen_training.progress.config import Config, parse_config
from qwen_training.progress.evaluate import evaluate, pause_test, post_press_events, press_summary, press_test, sync
from qwen_training.progress.loss import autocast, change_loss, event_loss, sample_loss
from qwen_training.progress.stream import CNT, FRAME, PREFIX, STG, P, StreamState, Tokens, forward_chunk, visibility
from qwen_training.progress.train import EpochSamples, TrainStream, fit_count_norm, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32   # fla 的 GDN kernel 只支持 bf16
WIDTH = 64


def tiny_backbone():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    text = dict(vocab_size=300, hidden_size=WIDTH, intermediate_size=128, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_conv_kernel_dim=4,
                linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"],
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0,
                                 "mrope_section": [3, 3, 2], "mrope_interleaved": True})
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16, spatial_merge_size=2,
                  temporal_patch_size=2, out_hidden_size=WIDTH, num_position_embeddings=64)
    config = Qwen3_5Config(text_config=text, vision_config=vision, vision_start_token_id=290,
                           vision_end_token_id=291, image_token_id=292)
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(config).to(DEVICE, DTYPE).eval()


class Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) % 250 for char in text]}


def image_processor():
    from transformers import Qwen2VLImageProcessorFast

    return Qwen2VLImageProcessorFast(patch_size=16, temporal_patch_size=2, merge_size=2,
                                     size={"shortest_edge": 32 * 32, "longest_edge": 128 * 128},
                                     image_mean=[0.5] * 3, image_std=[0.5] * 3)


def tiny_cfg(**overrides):
    # 关键帧相关的测试沿用标注关键帧；默认值 none 在 ConfigTests 里单独检查
    values = dict(head_hidden=32, lora_rank=4, window=3, max_keyframes=2, segment=4, device=DEVICE, accumulate=1,
                  iterations=2, log_every=1, eval_every=2, eval_episodes=1, save_every=100, workers=0,
                  keyframes="label", pause_prob=0.0, speed_prob=0.0)
    return Config(**{**values, **overrides})


def meta(kinds, frames, start, keep=False):
    n = len(kinds)
    return Tokens(kind=torch.tensor(kinds, device=DEVICE), frame=torch.tensor(frames, device=DEVICE),
                  index=torch.arange(start, start + n, device=DEVICE), keep=torch.full((n,), keep, device=DEVICE),
                  key=torch.zeros(n, dtype=torch.bool, device=DEVICE))


def positions(start, n):
    return torch.arange(start, start + n, device=DEVICE).expand(3, 1, -1)


def views(rng, count=3, size=64):
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(count)]


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backbone = tiny_backbone()
        cls.language = cls.backbone.model.language_model

    def test_chunked_stream_matches_full_forward(self):
        torch.manual_seed(1)
        embeds = torch.randn(1, 20, WIDTH, device=DEVICE, dtype=DTYPE)
        with torch.no_grad():
            full = self.language(inputs_embeds=embeds, position_ids=positions(0, 20)).last_hidden_state
            state = StreamState()
            first = forward_chunk(self.language, embeds[:, :12], positions(0, 12), meta([PREFIX] * 12, [-1] * 12, 0),
                                  state, window=100)
            second = forward_chunk(self.language, embeds[:, 12:17], positions(12, 5), meta([FRAME] * 5, [0] * 5, 12),
                                   state, window=100)
            third = forward_chunk(self.language, embeds[:, 17:], positions(17, 3), meta([FRAME] * 3, [1] * 3, 17),
                                  state, window=100)
        torch.testing.assert_close(torch.cat([first, second, third], 1), full, atol=5e-2, rtol=5e-2)

    def run_two_steps(self, readout_embeds, isolate_gdn=True):
        """前缀 4 个 token，之后两步，每步 5 个画面 token + STG·CNT·P；返回第二步的 hidden。"""
        torch.manual_seed(2)
        prefix = torch.randn(1, 4, WIDTH, device=DEVICE, dtype=DTYPE)
        frames = torch.randn(2, 5, WIDTH, device=DEVICE, dtype=DTYPE)
        state = StreamState()
        with torch.no_grad():
            forward_chunk(self.language, prefix, positions(0, 4), meta([PREFIX] * 4, [-1] * 4, 0), state, 3)
            for t in range(2):
                embeds = torch.cat([frames[t], readout_embeds[t]])[None]
                kinds = [FRAME] * 5 + [STG, CNT, P]
                hidden = forward_chunk(self.language, embeds, positions(4 + 8 * t, 8),
                                       meta(kinds, [t] * 8, 4 + 8 * t, keep=t == 0), state, 3, isolate_gdn)
        return hidden[0]

    def test_readouts_do_not_leak_into_video_or_stage(self):
        torch.manual_seed(3)
        readouts = torch.randn(2, 3, WIDTH, device=DEVICE, dtype=DTYPE)
        changed = readouts.clone()
        changed[0] = torch.randn(3, WIDTH, device=DEVICE, dtype=DTYPE)      # 只改第一步的三个 readout
        base, other = self.run_two_steps(readouts), self.run_two_steps(changed)
        torch.testing.assert_close(other, base)                 # 三个 readout 都不继承第一步的 readout
        leaky, leaky_other = self.run_two_steps(readouts, False), self.run_two_steps(changed, False)
        self.assertFalse(torch.allclose(leaky[:5], leaky_other[:5]))   # 不隔离时经 GDN 漏给画面

    def test_write_strength_scales_state_updates(self):
        torch.manual_seed(5)
        prefix = torch.randn(1, 4, WIDTH, device=DEVICE, dtype=DTYPE)
        frame = torch.randn(1, 5, WIDTH, device=DEVICE, dtype=DTYPE)

        def write(strength):
            state = StreamState()
            with torch.no_grad():
                forward_chunk(self.language, prefix, positions(0, 4), meta([PREFIX] * 4, [-1] * 4, 0), state, 3)
                before = {layer: value.clone() for layer, value in state.recurrent.items()}
                forward_chunk(self.language, frame, positions(4, 5), meta([FRAME] * 5, [0] * 5, 4), state, 3,
                              strength=strength)
            return before, state.recurrent

        before, closed = write(torch.zeros(5, device=DEVICE))
        for layer, value in before.items():
            torch.testing.assert_close(closed[layer], value)                     # 不写入也不衰减
        _, full = write(None)
        _, ones = write(torch.ones(5, device=DEVICE))
        for layer in full:
            torch.testing.assert_close(ones[layer], full[layer])
            self.assertFalse(torch.allclose(full[layer], before[layer]))

    def test_count_does_not_see_stage(self):
        torch.manual_seed(4)
        readouts = torch.randn(2, 3, WIDTH, device=DEVICE, dtype=DTYPE)
        changed = readouts.clone()
        changed[:, 0] = torch.randn(2, WIDTH, device=DEVICE, dtype=DTYPE)    # 改两步的 [STG]
        base, other = self.run_two_steps(readouts), self.run_two_steps(changed)
        torch.testing.assert_close(other[6], base[6])
        self.assertFalse(torch.allclose(other[7], base[7]))      # [P] 能看到 [STG]

    def test_visibility_table(self):
        keys = Tokens(kind=torch.tensor([PREFIX, FRAME, FRAME, STG, CNT, P, FRAME]),
                      frame=torch.tensor([-1, 0, 1, 4, 4, 4, 5]), index=torch.arange(7),
                      keep=torch.tensor([False, True, False, False, False, False, False]),
                      key=torch.tensor([False, False, True, False, False, False, False]))
        query = Tokens(kind=torch.tensor([FRAME, STG, CNT, P]), frame=torch.full((4,), 5),
                       index=torch.arange(7, 11), keep=torch.zeros(4, dtype=torch.bool),
                       key=torch.zeros(4, dtype=torch.bool))
        mask = visibility(query, keys, window=2)
        expected = [[1, 1, 0, 0, 0, 0, 1],    # 画面：前缀、首帧、窗口内画面
                    [1, 1, 0, 0, 0, 0, 1],    # [STG]：不看关键帧和任何 readout
                    [1, 1, 1, 0, 0, 0, 1],    # [CNT]：加关键帧，默认不看之前的 [CNT]
                    [1, 1, 1, 0, 0, 0, 1]]    # [P]：不看任何历史 readout
        self.assertEqual(mask.int().tolist(), expected)

    def test_progress_reads_count_only_removes_p_to_cnt_visibility(self):
        keys = meta([PREFIX, FRAME, STG, CNT, P, FRAME, STG, CNT, P], [-1, 0, 0, 0, 0, 1, 1, 1, 1], 0)
        query = keys.select(torch.tensor([False] * 5 + [True] * 4, device=DEVICE))
        baseline = visibility(query, keys, 3, progress_reads_history=True)
        ablated = visibility(query, keys, 3, progress_reads_count=False, progress_reads_history=True)
        removed = (query.kind[:, None] == P) & (keys.kind[None] == CNT)
        self.assertEqual(baseline[removed].sum().item(), 2)  # 当前和历史 CNT
        torch.testing.assert_close(ablated, baseline & ~removed)

    def test_count_reads_history_only_removes_cnt_to_past_cnt_visibility(self):
        keys = meta([PREFIX, FRAME, STG, CNT, P, FRAME, STG, CNT, P], [-1, 0, 0, 0, 0, 1, 1, 1, 1], 0)
        query = keys.select(torch.tensor([False] * 5 + [True] * 4, device=DEVICE))
        baseline = visibility(query, keys, 3, count_reads_history=True, progress_reads_history=True)
        ablated = visibility(query, keys, 3, count_reads_history=False, progress_reads_history=True)
        removed = torch.zeros_like(baseline)
        removed[2, 3] = True                     # 第 1 帧的 [CNT] 看第 0 帧的 [CNT]；仍看自己，P 仍看 CNT
        self.assertTrue(baseline[2, 3] and baseline[2, 7] and ablated[2, 7] and ablated[3, 3])
        torch.testing.assert_close(ablated, baseline & ~removed)

    def test_stage_reads_history_only_adds_stg_to_past_stg_visibility(self):
        keys = meta([PREFIX, FRAME, STG, CNT, P, FRAME, STG, CNT, P], [-1, 0, 0, 0, 0, 1, 1, 1, 1], 0)
        query = keys.select(torch.tensor([False] * 5 + [True] * 4, device=DEVICE))
        baseline = visibility(query, keys, 3)
        enabled = visibility(query, keys, 3, stage_reads_history=True)
        added = torch.zeros_like(baseline)
        added[1, 2] = True                       # 第 1 帧的 [STG] 看第 0 帧的 [STG]；仍不看 CNT、P
        self.assertFalse(baseline[1, 2])
        torch.testing.assert_close(enabled, baseline | added)

    def test_p_reads_only_current_readouts_when_history_disabled(self):
        keys = meta([PREFIX, FRAME, STG, CNT, P, FRAME, STG, CNT, P], [-1, 0, 0, 0, 0, 1, 1, 1, 1], 0)
        query = keys.select(torch.tensor([False] * 5 + [True] * 4, device=DEVICE))
        baseline = visibility(query, keys, 3, progress_reads_history=True)
        ablated = visibility(query, keys, 3, progress_reads_history=False)
        removed = torch.zeros_like(baseline)
        removed[3, 2:5] = True                   # 屏蔽之前的 STG、CNT、P，保留当前步三项
        self.assertTrue(baseline[3, 2:5].all())
        self.assertTrue(ablated[3, 6:9].all())
        torch.testing.assert_close(ablated, baseline & ~removed)

    def test_evict_keeps_first_frame_keyframes_and_window(self):
        state = StreamState()
        state.meta = Tokens(kind=torch.tensor([FRAME, CNT] * 7), frame=torch.arange(7).repeat_interleave(2),
                            index=torch.arange(14), keep=torch.tensor([True] + [False] * 13),
                            key=torch.zeros(14, dtype=torch.bool))
        state.stream = {3: (torch.zeros(1, 1, 14, 2), torch.zeros(1, 1, 14, 2))}
        state.frame = 6
        state.meta.key |= (state.meta.kind == FRAME) & (state.meta.frame >= 1) & (state.meta.frame <= 4)
        state.keyframes, state.extra_keyframes = [1, 2, 3], [4]
        state.evict(window=2, max_keyframes=2)
        kept = list(zip(state.meta.frame.tolist(), state.meta.kind.tolist(), strict=True))
        # 真正的关键帧 1、2、3 超过上限也全部保留；随机加入的帧 4 被删；最近一帧 6（下一帧的窗口是 6、7）。
        self.assertEqual(kept, [(0, FRAME), (1, FRAME), (2, FRAME), (3, FRAME), (6, FRAME), (6, CNT)])
        self.assertEqual(state.stream[3][0].shape[2], 6)


def fake_sample(rng, frames=6, grad_from=2):
    return {"task": "fake", "instruction": "move blocks", "steps": ["pick", "place"],
            "demo": [{"step": 0, "views": views(rng)}, {"step": 1, "views": views(rng, 1)}],
            "frames": [views(rng) for _ in range(frames)], "grad_from": grad_from,
            "stage": torch.tensor([0, 0, 0, 1, 1, 1])[:frames],
            "count": (torch.tensor([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 1]])[:frames, :, None]
                      >= torch.arange(1, MAX_COUNT + 1)).float(),
            "progress": torch.linspace(0, 1, frames), "total": 2,
            "stage_mask": torch.ones(frames, dtype=torch.bool),
            "count_mask": torch.ones(frames, 2, MAX_COUNT, dtype=torch.bool),
            "fed": torch.tensor([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0]])[:frames],   # 上一帧的真值次数
            "event_mask": torch.ones(frames, 2, dtype=torch.bool),
            "keyframes": torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.bool)[:frames],
            "extra_keyframes": torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)[:frames]}


class FakeSet:
    def __init__(self):
        self.item = fake_sample(np.random.default_rng(0))

    def __len__(self):
        return 1

    def sample(self, index, cfg, rng=None, segment=None):
        return self.item if segment else {**self.item, "grad_from": 0}


class ModelTests(unittest.TestCase):
    def make(self, **overrides):
        from qwen_training.progress.model import ProgressModel

        return ProgressModel(tiny_backbone(), image_processor(), Tokenizer(), tiny_cfg(**overrides))

    def test_rollout_backward_reaches_trainable_parts(self):
        model = self.make()
        sample = fake_sample(np.random.default_rng(0))
        with autocast(model.cfg):
            outputs = model.rollout(sample, sample["grad_from"])
        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[0]["stage"].shape, (3,))     # J+1
        self.assertEqual(outputs[0]["count"].shape, (2, MAX_COUNT))
        probability = outputs[0]["count"].sigmoid()
        self.assertTrue((probability[:, 1:] <= probability[:, :-1]).all())   # 做了至少 r 次的概率随 r 递减
        loss, _ = sample_loss(model, outputs, sample, model.cfg)
        (loss + sum(((param - init) ** 2).sum() for param, init in model.gates)).backward()
        lora = model.language.layers[3].self_attn.q_proj.up.weight
        gate = model.language.layers[0].linear_attn.A_log
        heads = [param for name, param in model.named_parameters() if name.endswith("head.0.weight")]
        for param in (model.special, model.count_gaps, model.count_feedback[-1].weight, lora, gate, *heads):
            self.assertGreater(param.grad.abs().sum().item(), 0)
        self.assertIsNone(model.backbone.model.visual.patch_embed.proj.weight.grad)

    def test_count_feedback_reaches_cnt_only(self):
        """窗口外的计数概率回传影响下一步 CNT，但不影响 STG。"""
        model = self.make(window=2, keyframes="none", count_feedback=True)
        torch.nn.init.normal_(model.count_feedback[-1].weight, std=0.1)
        sample = fake_sample(np.random.default_rng(2))
        frames = [model.encode_images(views) for views in sample["frames"]]
        runs = []
        for noise in (0.0, 1.0):
            with torch.no_grad(), autocast(model.cfg), mock.patch.object(model, "encode_images", side_effect=frames):
                state = model.start(sample["instruction"], sample["steps"], [])
                outs = [model.step(state, None) for _ in range(5)]
                state.count_probs = (state.count_probs + noise).clamp(0, 1)
                outs.append(model.step(state, None))
            runs.append(outs)
        self.assertFalse(torch.allclose(runs[0][-1]["count"], runs[1][-1]["count"]))
        for clean, noisy in zip(runs[0], runs[1], strict=True):
            torch.testing.assert_close(clean["stage"], noisy["stage"])

    def test_feedback_ablation_zeros_probabilities_but_keeps_step_features(self):
        model = self.make(count_feedback=False)
        torch.nn.init.normal_(model.count_feedback[-1].weight, std=0.1)
        steps = torch.randn(2, WIDTH, device=DEVICE)
        probs = torch.rand(1, 2, MAX_COUNT, device=DEVICE)
        expected = model.special[2] + model.count_feedback(
            torch.cat([steps[None], torch.zeros_like(probs)], -1)).sum(1)
        torch.testing.assert_close(model._count_inputs(steps, probs), expected)
        torch.testing.assert_close(model._count_inputs(steps, probs + 1), expected)
        self.assertFalse(torch.allclose(model._count_inputs(steps + 1, probs), expected))
        model.cfg.count_feedback = True
        self.assertFalse(torch.allclose(model._count_inputs(steps, probs), expected))

    def test_p_ablation_changes_progress_without_changing_count_or_stage(self):
        model = self.make().eval()
        sample = fake_sample(np.random.default_rng(2))
        with torch.no_grad(), autocast(model.cfg):
            baseline = model.rollout(sample, keyframes="model")
            model.cfg.progress_reads_count = False
            ablated = model.rollout(sample, keyframes="model")
        for original, changed in zip(baseline, ablated, strict=True):
            for key in ("stage", "count"):
                torch.testing.assert_close(original[key], changed[key])
        self.assertTrue(any(not torch.allclose(a["progress"], b["progress"])
                            for a, b in zip(baseline, ablated, strict=True)))

    def test_count_history_ablation_reaches_rollout_with_feedback_disabled(self):
        model = self.make(count_feedback=False, count_reads_history=True).eval()
        sample = fake_sample(np.random.default_rng(2))
        with torch.no_grad(), autocast(model.cfg):
            baseline = model.rollout(sample, keyframes="none")
            model.cfg.count_reads_history = False
            ablated = model.rollout(sample, keyframes="none")
        for key in ("stage", "count", "progress"):
            torch.testing.assert_close(baseline[0][key], ablated[0][key])
        for first, second in zip(baseline, ablated, strict=True):
            torch.testing.assert_close(first["stage"], second["stage"])
        self.assertTrue(any(not torch.allclose(a["count"], b["count"])
                            for a, b in zip(baseline[1:], ablated[1:], strict=True)))

    def test_oracle_feedback_uses_previous_truth_without_current_label_leakage(self):
        for mode in ("level", "event"):
            with self.subTest(mode=mode):
                model = self.make(count_mode=mode, count_feedback=True, oracle_count_feedback=True).eval()
                sample = fake_sample(np.random.default_rng(2))
                inputs, original = [], model._count_inputs

                def capture(steps, probs, inputs=inputs, original=original):
                    inputs.append(probs.detach().cpu().clone())
                    return original(steps, probs)

                with torch.no_grad(), autocast(model.cfg), mock.patch.object(
                        model, "_count_inputs", side_effect=capture):
                    outputs = model.rollout(sample, keyframes="model")
                    for t, probs in enumerate(inputs):
                        expected = sample["count"][t - 1] if t else torch.zeros_like(sample["count"][0])
                        torch.testing.assert_close(probs[0], expected)
                        if mode == "event":
                            torch.testing.assert_close(outputs[t]["fed"].cpu(), expected.sum(-1).long())
                    changed = {**sample, "count": sample["count"].clone()}
                    changed["count"][-1] = 1 - changed["count"][-1]
                    other = model.rollout(changed, keyframes="model")
                for first, second in zip(outputs, other, strict=True):
                    for key in ("stage", "count", "progress"):
                        torch.testing.assert_close(first[key], second[key])

    def test_oracle_feedback_keeps_keyframes_based_on_predictions(self):
        model = self.make(count_feedback=True, oracle_count_feedback=True).eval()
        sample = fake_sample(np.random.default_rng(2))
        states, original = [], model.start
        with torch.no_grad(), autocast(model.cfg), mock.patch.object(
                model, "start", side_effect=lambda *args: states.append(original(*args)) or states[-1]):
            outputs = model.rollout(sample, keyframes="model")
        probs = torch.stack([out["count"] for out in outputs]).float().sigmoid()
        expected = ((probs[1:] - probs[:-1]).abs().flatten(1).amax(-1) > model.cfg.delta).nonzero().flatten() + 1
        self.assertEqual(states[0].keyframes, expected.tolist())

    def test_checkpoint_rejects_incomplete_or_obsolete_trainable_weights(self):
        model = self.make()
        weights = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}
        model.load_trainable(weights)
        removed = weights.pop("count_feedback.2.weight")
        with self.assertRaisesRegex(ValueError, "count_feedback.2.weight"):
            model.load_trainable(weights)
        weights["count_feedback.2.weight"] = removed
        weights["feedback.weight"] = removed
        with self.assertRaisesRegex(ValueError, "feedback.weight"):
            model.load_trainable(weights)

    def test_count_norm_and_event_loss_train_and_checkpoint(self):
        """计数头输入标准化：统计量训练前汇总、训练时更新、验证时固定，存进 checkpoint；事件损失能反传到计数头。"""
        model = self.make(count_norm=True)
        sample = fake_sample(np.random.default_rng(0))
        fit_count_norm(model, [sample, fake_sample(np.random.default_rng(1))], model.cfg)
        # 前半是 h_CNT，在一步的各项间完全相同；方差跨步汇总，每一维都不为 0。
        self.assertTrue(model.count_norm.initialized)
        self.assertTrue((model.count_norm.var > 0).all())
        fitted = model.count_norm.mean.clone()
        with autocast(model.cfg):
            outputs = model.rollout(sample, sample["grad_from"])
            loss, parts = sample_loss(model, outputs, sample, model.cfg, event_weight=1.0)
        loss.backward()
        self.assertGreater(parts["event"], 0)
        self.assertGreater(model.count_head[0].weight.grad.abs().sum().item(), 0)
        self.assertFalse(torch.equal(model.count_norm.mean, fitted))   # 训练时滑动更新
        state = model.trainable_state()
        self.assertIn("count_norm.square", state)
        model.eval()
        mean = model.count_norm.mean.clone()
        with torch.no_grad(), autocast(model.cfg):
            model.rollout(sample)
        torch.testing.assert_close(model.count_norm.mean, mean)
        fresh = self.make(count_norm=True)
        fresh.load_trainable(state)
        torch.testing.assert_close(fresh.count_norm.var, model.count_norm.var)
        with self.assertRaisesRegex(ValueError, "count_norm.mean"):
            self.make().load_trainable(state)

    def test_evaluate_without_keyframes_keeps_default_best_metric(self):
        model = self.make(keyframes="none")
        result = evaluate(model, FakeSet(), model.cfg, 1)
        self.assertIn(model.cfg.best_metric, result["overall"])
        self.assertFalse(any(key.startswith("val/oracle/") for key in result["overall"]))

    def test_teacher_forcing_in_parallel_matches_stepwise(self):
        """覆盖热身、零热身与段内随机帧淘汰；回传头非零时仍与逐步 teacher forcing 一致。"""
        cases = [(3, 2, [2, 5], [1]), (0, 1, [4], [1, 3]), (0, 2, [2, 5], [])]
        for grad_from, limit, real_frames, extra_frames in cases:
            with self.subTest(grad_from=grad_from, max_keyframes=limit, extra=extra_frames):
                model = self.make(window=2, max_keyframes=limit)
                torch.nn.init.normal_(model.count_feedback[-1].weight, std=0.1)
                sample = fake_sample(np.random.default_rng(3), grad_from=grad_from)
                sample["keyframes"].zero_()
                sample["keyframes"][real_frames] = True
                sample["extra_keyframes"].zero_()
                sample["extra_keyframes"][extra_frames] = True
                truth = sample["count"].to(DEVICE)
                with torch.no_grad(), autocast(model.cfg):
                    parallel = model.rollout(sample, grad_from, teacher=True)
                    state = model.start(sample["instruction"], sample["steps"], sample["demo"])
                    stepwise = []
                    for t, views in enumerate(sample["frames"]):
                        state.count_probs = truth[t - 1] if t else torch.zeros_like(truth[0])
                        stepwise.append(model.step(state, views))
                        real, extra = sample["keyframes"][t], sample["extra_keyframes"][t]
                        if real or extra:
                            state.meta.key |= (state.meta.frame == t) & (state.meta.kind == FRAME)
                            (state.keyframes if real else state.extra_keyframes).append(t)
                        state.evict(model.cfg.window, model.cfg.max_keyframes)
                self.assertEqual(len(parallel), len(sample["frames"]) - grad_from)
                tolerance = 5e-2 if DEVICE == "cuda" else 2e-5
                for fast, slow in zip(parallel, stepwise[grad_from:], strict=True):
                    for name in ("stage", "count", "progress"):
                        torch.testing.assert_close(fast[name], slow[name], atol=tolerance, rtol=tolerance)

    def test_rollout_without_feedback_or_keyframes_runs_parallel_and_matches_stepwise(self):
        """chunk 小于或大于 window 时，真正的并行分支与逐帧输出、梯度均一致。"""
        for chunk_size, grad_from in ((1, 0), (1, 3), (4, 0), (4, 3), (8, 3)):
            with self.subTest(chunk_size=chunk_size, grad_from=grad_from):
                model = self.make(window=2, chunk_size=chunk_size, keyframes="none")
                torch.nn.init.normal_(model.count_feedback[-1].weight, std=0.1)
                sample = fake_sample(np.random.default_rng(5), grad_from=grad_from)
                with mock.patch.object(model, "teacher_steps", wraps=model.teacher_steps) as chunks, \
                        mock.patch.object(model, "encode_images", wraps=model.encode_images) as images, \
                        mock.patch.object(model, "_heads", wraps=model._heads) as heads, autocast(model.cfg):
                    parallel = model.rollout(sample, grad_from)
                    loss, _ = sample_loss(model, parallel, sample, model.cfg)
                expected = [(i, i + 1) for i in range(6)] if chunk_size == 1 else \
                    [(0, 4), (4, 6)] if grad_from == 0 else [(0, 3), (3, 6)]
                self.assertEqual([(call.args[2], call.args[3]) for call in chunks.call_args_list], expected)
                self.assertEqual(images.call_count, len(sample["demo"]) + chunks.call_count)
                self.assertTrue(all(call.args[1].ndim == 2 for call in heads.call_args_list))
                loss.backward()
                gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
                model.zero_grad(set_to_none=True)
                with autocast(model.cfg):
                    state = model.start(sample["instruction"], sample["steps"], sample["demo"])
                    stepwise = []
                    for t, views in enumerate(sample["frames"]):
                        with torch.set_grad_enabled(t >= grad_from):
                            out = model.step(state, views)
                        if t >= grad_from:
                            stepwise.append(out)
                        state.evict(model.cfg.window, model.cfg.max_keyframes)
                    slow_loss, _ = sample_loss(model, stepwise, sample, model.cfg)
                slow_loss.backward()
                tolerance = 5e-2 if DEVICE == "cuda" else 2e-5
                for fast, slow in zip(parallel, stepwise, strict=True):
                    for name in ("stage", "count", "progress"):
                        torch.testing.assert_close(fast[name], slow[name], atol=tolerance, rtol=tolerance)
                for name, param in model.named_parameters():
                    if name in gradients:
                        torch.testing.assert_close(param.grad, gradients[name], atol=tolerance, rtol=tolerance)
                self.assertEqual(state.keyframes, [])

    def test_chunk_24_realigns_training_after_partial_warmup_chunk(self):
        from qwen_training.progress.evaluate import take

        model = self.make(window=15, chunk_size=24, keyframes="none")
        sample = take(fake_sample(np.random.default_rng(0), grad_from=0), list(range(6)) * 9)
        sample["grad_from"] = 30
        calls = []
        original = model.teacher_steps

        def capture(state, sample, start, end, keyframes):
            calls.append((start, end, torch.is_grad_enabled()))
            return original(state, sample, start, end, keyframes)

        with mock.patch.object(model, "teacher_steps", side_effect=capture), autocast(model.cfg):
            outputs = model.rollout(sample, grad_from=30)
            loss, _ = sample_loss(model, outputs, sample, model.cfg)
        self.assertEqual(calls, [(0, 24, False), (24, 30, False), (30, 54, True)])
        self.assertEqual(len(outputs), 24)
        loss.backward()
        self.assertGreater(model.count_head[0].weight.grad.abs().sum().item(), 0)
        self.assertEqual((Config().chunk_size, Config().window), (24, 15))
        for name in ("autodl_rmbench_4b.yaml", "autodl_rmbench_4b_gates.yaml"):
            cfg = parse_config(["--config", str(Path(__file__).parents[1] / "qwen_training/progress/configs" / name)])
            self.assertEqual((cfg.chunk_size, cfg.window, cfg.segment), (24, 15, 24))

    def test_activation_checkpointing_preserves_outputs_gradients_and_norm_updates(self):
        for keyframes in ("none", "model"):
            with self.subTest(keyframes=keyframes):
                model = self.make(window=2, chunk_size=2, keyframes=keyframes, count_norm=True)
                model.count_norm.fit(torch.zeros_like(model.count_norm.mean),
                                     torch.ones_like(model.count_norm.square))
                initial = {n: t.clone() for n, t in model.state_dict().items()}
                sample = fake_sample(np.random.default_rng(5))
                with autocast(model.cfg):
                    reference = model.rollout(sample, grad_from=2)
                    loss, _ = sample_loss(model, reference, sample, model.cfg)
                loss.backward()
                gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
                mean, square = model.count_norm.mean.clone(), model.count_norm.square.clone()
                model.zero_grad(set_to_none=True)
                model.load_state_dict(initial)
                model.cfg.activation_checkpointing = True
                with autocast(model.cfg):
                    outputs = model.rollout(sample, grad_from=2)
                    loss, _ = sample_loss(model, outputs, sample, model.cfg)
                loss.backward()
                tolerance = 5e-2 if DEVICE == "cuda" else 2e-5
                for a, b in zip(reference, outputs, strict=True):
                    for name in ("stage", "count", "progress"):
                        torch.testing.assert_close(a[name], b[name], atol=tolerance, rtol=tolerance)
                self.assertEqual({n for n, p in model.named_parameters() if p.grad is not None}, set(gradients))
                for name, param in model.named_parameters():
                    if name in gradients:
                        torch.testing.assert_close(param.grad, gradients[name], atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(model.count_norm.mean, mean)
                torch.testing.assert_close(model.count_norm.square, square)

    def test_separate_warmup_and_training_chunks_preserve_outputs_and_gradients(self):
        from qwen_training.progress.evaluate import take

        model = self.make(window=3, chunk_size=4, keyframes="none")
        sample = take(fake_sample(np.random.default_rng(5), grad_from=0), list(range(6)) * 2)
        sample["grad_from"] = 8
        with autocast(model.cfg):
            reference = model.rollout(sample, grad_from=8)
            loss, _ = sample_loss(model, reference, sample, model.cfg)
        loss.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        model.cfg.warmup_chunk_size, model.cfg.train_chunk_size = 8, 1
        with mock.patch.object(model, "teacher_steps", wraps=model.teacher_steps) as chunks, autocast(model.cfg):
            outputs = model.rollout(sample, grad_from=8)
            loss, _ = sample_loss(model, outputs, sample, model.cfg)
        self.assertEqual([(c.args[2], c.args[3]) for c in chunks.call_args_list],
                         [(0, 8), (8, 9), (9, 10), (10, 11), (11, 12)])
        loss.backward()
        tolerance = 5e-2 if DEVICE == "cuda" else 2e-5
        for a, b in zip(reference, outputs, strict=True):
            for name in ("stage", "count", "progress"):
                torch.testing.assert_close(a[name], b[name], atol=tolerance, rtol=tolerance)
        for name, param in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(param.grad, gradients[name], atol=tolerance, rtol=tolerance)
        with mock.patch.object(model, "teacher_steps", wraps=model.teacher_steps) as chunks, torch.no_grad(), \
                autocast(model.cfg):
            model.rollout(sample)   # 评估：不求梯度，全程按热身的 chunk 切
        self.assertEqual([(c.args[2], c.args[3]) for c in chunks.call_args_list], [(0, 8), (8, 12)])
        for name in ("warmup_chunk_size", "train_chunk_size"):
            setattr(model.cfg, name, -1)
            with self.assertRaisesRegex(ValueError, "chunk_size"):
                model.rollout(sample, grad_from=8)
            setattr(model.cfg, name, 0)

    def test_batched_heads_preserve_running_normalization_and_gradients(self):
        import copy

        for mode in ("level", "event"):
            for training in (False, True):
                with self.subTest(mode=mode, training=training):
                    model = self.make(count_mode=mode, count_norm=True)
                    model.count_norm.fit(torch.zeros(2 * WIDTH, device=DEVICE),
                                         torch.ones(2 * WIDTH, device=DEVICE))
                    model.train(training)
                    slow = copy.deepcopy(model)
                    steps = torch.randn(2, WIDTH, device=DEVICE)
                    hidden = torch.randn(4, 3, WIDTH, device=DEVICE)
                    with autocast(model.cfg):
                        batched = model._heads(steps, *hidden.unbind(1))
                        sequential = [slow._heads(steps, *frame.unbind(0)) for frame in hidden]
                    sum(value.sum() for value in batched.values()).backward()
                    sum(value.sum() for out in sequential for value in out.values()).backward()
                    tolerance = 5e-2 if DEVICE == "cuda" else 2e-5
                    for name, values in batched.items():
                        torch.testing.assert_close(values, torch.stack([out[name] for out in sequential]),
                                                   atol=tolerance, rtol=tolerance)
                    for (name, param), (_, other) in zip(model.named_parameters(), slow.named_parameters(), strict=True):
                        if param.grad is not None:
                            torch.testing.assert_close(param.grad, other.grad, atol=tolerance, rtol=tolerance, msg=name)
                    torch.testing.assert_close(model.count_norm.mean, slow.count_norm.mean)
                    torch.testing.assert_close(model.count_norm.square, slow.count_norm.square)

    def test_chunk_size_is_validated_and_independent_of_window(self):
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            self.make(chunk_size=0).rollout(fake_sample(np.random.default_rng(0)))
        cfg = parse_config(["--chunk-size", "24"])
        self.assertEqual((cfg.chunk_size, cfg.window), (24, 15))

    def test_event_mode_accumulates_counts_and_teacher_matches_stepwise(self):
        """event：次数在模型外累加并回传；把逐步推理时实际回传的次数作为 fed 给 teacher forcing，两者输出一致。"""
        model = self.make(count_mode="event", window=2)
        torch.nn.init.normal_(model.count_feedback[-1].weight, std=0.1)
        torch.nn.init.normal_(model.count_head[-1].weight, std=1.0)   # 让预测的变化不全是 0
        sample = fake_sample(np.random.default_rng(3), grad_from=1)
        with torch.no_grad(), autocast(model.cfg):
            stepwise = model.rollout(sample)
            fed = torch.stack([out["fed"] for out in stepwise])
            changes = torch.stack([out["count"].argmax(-1) - 1 for out in stepwise])
            torch.testing.assert_close(fed[1:], (fed[:-1] + changes[:-1]).clamp(0, MAX_COUNT))
            self.assertEqual(fed[0].tolist(), [0, 0])
            self.assertEqual(stepwise[0]["count"].shape, (2, 3))
            parallel = model.rollout({**sample, "fed": fed.cpu()}, 1, teacher=True)
        tolerance = 5e-2 if DEVICE == "cuda" else 2e-5
        for fast, slow in zip(parallel, stepwise[1:], strict=True):
            for name in ("stage", "count", "progress"):
                torch.testing.assert_close(fast[name], slow[name], atol=tolerance, rtol=tolerance)
            self.assertEqual(fast["fed"].tolist(), slow["fed"].tolist())

    def test_event_mode_trains_and_evaluates(self):
        model = self.make(count_mode="event", teacher=True, iterations=2, eval_every=2)
        with tempfile.TemporaryDirectory() as out:
            model.cfg.out = out
            history = train(model, model.cfg, FakeSet(), FakeSet())
            records = [json.loads(line) for line in (Path(out) / "log.jsonl").read_text().splitlines()]
        self.assertTrue(all(math.isfinite(r["count"]) for r in history))
        final = [r for r in records if "val/model/count_acc" in r][-1]
        self.assertAlmostEqual(final["val/model/count_zero_acc"], 7 / 12)    # fake_sample 12 个 (帧, 项) 里 7 个为 0

    def test_teacher_future_keyframe_does_not_change_past_outputs(self):
        model = self.make(window=2, max_keyframes=1)
        sample = fake_sample(np.random.default_rng(4), grad_from=0)
        sample["keyframes"].zero_()
        future = {**sample, "keyframes": sample["keyframes"].clone()}
        future["keyframes"][-1] = True
        with torch.no_grad(), autocast(model.cfg):
            original = model.rollout(sample, teacher=True)
            changed = model.rollout(future, teacher=True)
        for before, after in zip(original[:-1], changed[:-1], strict=True):
            for name in ("stage", "count", "progress"):
                torch.testing.assert_close(before[name], after[name])

    def test_teacher_rollout_backward_and_caps_extra_keyframes(self):
        model = self.make()
        sample = fake_sample(np.random.default_rng(0))             # 2 个标注关键帧 + 1 个随机帧，上限 2
        with autocast(model.cfg):
            outputs = model.rollout(sample, sample["grad_from"], teacher=True)
            loss, _ = sample_loss(model, outputs, sample, model.cfg)
        loss.backward()
        self.assertEqual(len(outputs), len(sample["frames"]) - sample["grad_from"])
        lora = model.language.layers[3].self_attn.q_proj.up.weight
        for param in (model.special, model.count_feedback[-1].weight, lora):
            self.assertGreater(param.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            model.rollout(sample, keyframes="model", teacher=True)

    def test_model_triggered_keyframes_run(self):
        model = self.make(keyframes="model", delta=0.0, window=2)
        with torch.no_grad(), autocast(model.cfg):
            self.assertEqual(len(model.rollout(fake_sample(np.random.default_rng(1)))), 6)

    def test_train_loop_overfits_evaluates_and_resumes(self):
        settings = dict(iterations=12, head_lr=3e-3, lr=3e-3, warmup=2, eval_every=12, save_every=6)
        model = self.make(**settings)
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as again:
            model.cfg.out = out
            history = train(model, model.cfg, FakeSet(), FakeSet())
            self.assertLess(history[-1]["loss"], history[0]["loss"])
            records = [json.loads(line) for line in (Path(out) / "log.jsonl").read_text().splitlines()]
            self.assertEqual([r["iteration"] for r in records if "val/model/count_acc" in r], [0, 12])   # 训练前、结束时
            for name in ("config.yaml", "latest.pt", "best.pt", "step6.pt", "step12.pt", "eval/iter0.json"):
                self.assertTrue((Path(out) / name).exists(), name)
            result = json.loads((Path(out) / "eval" / "iter12.json").read_text())
            self.assertTrue(0 <= result["per_task"]["fake"]["model/progress_mae"] <= 1)
            self.assertTrue(any(name.startswith("events") for name in os.listdir(Path(out) / "tensorboard")))

            # 从 step6.pt 继续：迭代号和学习率调度都接着第 7 次走。
            resumed = self.make(**settings, resume=str(Path(out) / "step6.pt"))
            resumed.cfg.out = again
            continued = train(resumed, resumed.cfg, FakeSet(), FakeSet())
            self.assertEqual(continued[0]["iteration"], 7)
            self.assertAlmostEqual(continued[0]["lr_head"], history[6]["lr_head"])
            self.assertAlmostEqual(continued[0]["lr_gate"], history[6]["lr_gate"])

    def test_train_fits_count_norm_before_training(self):
        model = self.make(count_norm=True, count_norm_samples=2, iterations=1, eval_at_start=False)
        with tempfile.TemporaryDirectory() as out, mock.patch.object(
                model.count_norm, "fit", wraps=model.count_norm.fit) as fit:
            model.cfg.out = out
            train(model, model.cfg, FakeSet())
        fit.assert_called_once()
        self.assertTrue(model.count_norm.initialized)
        # FakeSet 重复同一样本；bf16 下恒定维度可合法地有零方差，归一化已用 epsilon 保护。
        self.assertTrue(torch.isfinite(model.count_norm.var).all())
        self.assertTrue((model.count_norm.var >= 0).all())
        self.assertTrue((model.count_norm.var > 0).any())

    def test_image_resize_bounds_preserve_views_and_do_not_upscale(self):
        from qwen_training.progress.model import resize_images
        images = [np.zeros((480, 640, 3), np.uint8), np.zeros((768, 960, 3), np.uint8),
                  np.zeros((120, 160, 3), np.uint8)]
        self.assertIs(resize_images(images, 0, 0), images)
        resized = resize_images(images, 320, 240)
        self.assertEqual([x.shape for x in resized], [(240, 320, 3), (240, 300, 3), (120, 160, 3)])
        self.assertEqual(images[0].shape, (480, 640, 3))
        self.assertIs(resized[-1], images[-1])
        with self.assertRaises(ValueError):
            resize_images(images, 320, 0)
        model = self.make(image_max_width=32, image_max_height=32)
        with mock.patch.object(model, "image_processor", wraps=model.image_processor) as processor:
            model.encode_images(images)
        self.assertTrue(all(x.shape[0] <= 32 and x.shape[1] <= 32
                            for x in processor.call_args.kwargs["images"]))

    def test_multisegment_detaches_cached_prefix_and_matches_forward(self):
        from qwen_training.progress.evaluate import take
        model = self.make(keyframes="none", count_feedback=False, chunk_size=2,
                          train_chunk_size=2, warmup_chunk_size=2, activation_checkpointing=True)
        sample = take(fake_sample(np.random.default_rng(5)), list(range(6)) * 2)
        sample["segment_ranges"] = [(2, 4), (6, 8), (8, 10), (10, 12)]
        states = []
        original = model.start
        def capture(*args):
            state = original(*args)
            states.append(state)
            return state
        with mock.patch.object(model, "start", side_effect=capture), autocast(model.cfg):
            iterator = model.rollout_segments(sample)
            outputs, lo, hi = next(iterator)
            self.assertTrue(states[0].steps.requires_grad)
            loss = sum(out["count"].sum() for out in outputs)
            loss.backward()
            del outputs, loss
            for outputs, lo, hi in iterator:
                self.assertFalse(states[0].steps.requires_grad)
                self.assertTrue(all(not k.requires_grad and not v.requires_grad
                                    for k, v in states[0].prefix.values()))
                labels = take(sample, list(range(lo, hi)))
                labels["grad_from"] = 0
                loss, _ = sample_loss(model, outputs, labels, model.cfg)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                del outputs, loss
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertFalse(states[0].steps.requires_grad)

    def test_multisegment_epoch_budget_normalization_and_resume(self):
        class MultiSet(FakeSet):
            episodes = (("put_back_block", 9), ("put_back_block", 10), ("put_back_block", 11))
            def sample(self, index, cfg, rng=None, segment=None):
                sample = super().sample(index, cfg, rng, segment)
                sample["segment_ranges"] = [(0, 1), (1, 2), (2, 3), (3, 4)]
                return sample
        settings = dict(epochs=3, max_updates=3, segments_per_trajectory=4, accumulate=8,
                        keyframes="none", count_feedback=False, save_every=1, eval_at_start=False)
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as again:
            model = self.make(**settings, out=out)
            history = train(model, model.cfg, MultiSet())
            self.assertEqual([(r["epoch"], r["samples"], r["segments"]) for r in history],
                             [(1, 2, 8), (1, 1, 4), (2, 2, 8)])
            restored = self.make(**settings, out=again, resume=str(Path(out) / "step1.pt"))
            continuation = train(restored, restored.cfg, MultiSet())
            self.assertEqual([r["iteration"] for r in continuation], [2, 3])
            for full, resumed in zip(history[1:], continuation, strict=True):
                self.assertAlmostEqual(full["loss"], resumed["loss"], places=5)
            for name, param in model.trainable_state().items():
                torch.testing.assert_close(param, restored.trainable_state()[name])
        ranges = data.segment_ranges(40, 24, 4, [2, 12, 22, 32], 1, random.Random(0))
        self.assertEqual(ranges, [(0, 10), (10, 20), (20, 30), (30, 40)])
        ranges = data.segment_ranges(300, 24, 4, [20, 100, 180, 260], 1, random.Random(0))
        self.assertEqual(len(ranges), 4)
        self.assertTrue(all(b - a == 24 for a, b in ranges))
        self.assertTrue(all(b <= c for (_, b), (c, _) in pairwise(ranges)))

    def test_epoch_training_covers_tail_and_resumes_mid_epoch(self):
        class ThreeEpisodes(FakeSet):
            episodes = (("put_back_block", 9), ("put_back_block", 10), ("put_back_block", 11))

            def __init__(self):
                super().__init__()
                self.visits = []

            def sample(self, index, cfg, rng=None, segment=None):
                self.visits.append(index)
                return super().sample(index, cfg, rng, segment)

        settings = dict(epochs=2, accumulate=2, warmup=2, save_every=1, eval_at_start=False)
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as again:
            model = self.make(**settings, out=out)
            dataset = ThreeEpisodes()
            history = train(model, model.cfg, dataset)
            self.assertEqual(len(history), 4)
            self.assertEqual([(r["epoch"], r["samples"]) for r in history], [(1, 2), (1, 1), (2, 2), (2, 1)])
            self.assertEqual(Counter(dataset.visits), {0: 2, 1: 2, 2: 2})
            resumed = self.make(**settings, out=again, resume=str(Path(out) / "step1.pt"))
            continued = train(resumed, resumed.cfg, ThreeEpisodes())
            self.assertEqual([r["iteration"] for r in continued], [2, 3, 4])
            for full, restored in zip(history[1:], continued, strict=True):
                self.assertAlmostEqual(full["loss"], restored["loss"], places=5)
                self.assertEqual(full["lr_head"], restored["lr_head"])
            for name, param in model.trainable_state().items():
                torch.testing.assert_close(param, resumed.trainable_state()[name])

    def test_epoch_tail_uses_actual_accumulation_divisor(self):
        class ThreeEpisodes(FakeSet):
            episodes = (("put_back_block", 9), ("put_back_block", 10), ("put_back_block", 11))

        with tempfile.TemporaryDirectory() as out:
            model = self.make(out=out, epochs=1, accumulate=2, head_lr=0, lr=0, gate_lr=0,
                              gate_l2=0, eval_at_start=False)
            history = train(model, model.cfg, ThreeEpisodes())
            self.assertAlmostEqual(history[0]["loss"], history[1]["loss"], places=5)
            self.assertAlmostEqual(history[0]["grad_norm"], history[1]["grad_norm"], places=4)

    def test_resume_and_init_from_are_mutually_exclusive(self):
        model = self.make(resume="resume.pt", init_from="stage1.pt")
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            train(model, model.cfg, FakeSet())

    def test_stage2_starts_from_stage1_checkpoint(self):
        """stage 1 用 teacher forcing 训练；stage 2 只载入它的模型参数，迭代号和学习率调度从头开始。"""
        stage1 = self.make(teacher=True, iterations=2, save_every=2)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            stage1.cfg.out = first
            train(stage1, stage1.cfg, FakeSet())
            trained = torch.load(Path(first) / "step2.pt", map_location="cpu", weights_only=False)["model"]
            stage2 = self.make(iterations=2, init_from=str(Path(first) / "step2.pt"))
            stage2.cfg.out = second
            with mock.patch.object(stage2, "rollout", wraps=stage2.rollout) as rollout, \
                    mock.patch.object(stage2, "load_trainable", wraps=stage2.load_trainable) as load:
                history = train(stage2, stage2.cfg, FakeSet())
        torch.testing.assert_close(load.call_args.args[0], trained)
        self.assertTrue(all(call.kwargs["teacher"] is False for call in rollout.call_args_list))
        self.assertEqual(history[0]["iteration"], 1)
        self.assertLess(history[0]["lr_head"], stage2.cfg.head_lr)                       # 预热重新开始

    def test_main_seeds_before_building_model(self):
        """同一个 seed 两次运行 main，新加部分的初始化完全相同；换 seed 则不同。"""
        from qwen_training.progress import train as train_module
        from qwen_training.progress.model import ProgressModel

        def initial(seed):
            backbone = tiny_backbone()      # 它自己会设 seed，所以在 main 之前建好，main 里只剩新加部分的初始化
            parts = backbone, image_processor(), Tokenizer(), tiny_cfg()
            with mock.patch.object(ProgressModel, "from_pretrained", side_effect=lambda *args: ProgressModel(*parts)), \
                    mock.patch.object(data, "RMBench"), mock.patch.object(train_module, "train") as run:
                train_module.main(["--seed", str(seed)])
            model = run.call_args.args[0]
            return {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

        first, again, other = initial(0), initial(0), initial(1)
        for name in first:
            torch.testing.assert_close(first[name], again[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["special"], other["special"]))

    def test_export_for_viewer(self):
        from qwen_training.progress.visualize import export

        model = self.make()
        sample = fake_sample(np.random.default_rng(0))
        result = json.loads(json.dumps(export(model, sample, "label", fps=5)))       # 能写成 JSON
        self.assertEqual(result["count_mode"], "level")
        self.assertEqual(len(result["frames"]), len(sample["frames"]))
        self.assertEqual([frame["keyframe"] for frame in result["frames"]], sample["keyframes"].tolist())
        last = result["frames"][-1]
        self.assertEqual(last["truth"], {"stage": 1, "count": [1, 1], "progress": 1.0})
        self.assertEqual(len(last["prediction"]["stage_probs"]), 3)                          # J + 1
        self.assertEqual(len(last["prediction"]["count_probs"][0]), MAX_COUNT)
        self.assertEqual(len(last["images"]), 3)
        self.assertTrue(last["images"][0].startswith("data:image/jpeg;base64,"))
        self.assertAlmostEqual(result["frames"][1]["time"], 0.2)

    def test_export_event_mode_matches_rollout(self):
        """event：导出的次数是实际累加的次数，模型关键帧与 rollout 实际标记的一致。"""
        from qwen_training.progress.evaluate import predicted_counts
        from qwen_training.progress.visualize import export

        model = self.make(count_mode="event")
        torch.nn.init.constant_(model.count_head[-1].bias[2], 5.0)       # 每一步都预测 +1：次数 1 → 6
        sample = fake_sample(np.random.default_rng(0))
        states, start = [], model.start
        with mock.patch.object(model, "start", side_effect=lambda *args: states.append(start(*args)) or states[-1]):
            result = json.loads(json.dumps(export(model, sample, "model", fps=5)))
        model.eval()
        with torch.no_grad(), autocast(model.cfg):
            counts = predicted_counts(model, model.rollout(sample, keyframes="model"))
        self.assertEqual([frame["prediction"]["count"] for frame in result["frames"]], counts.tolist())
        self.assertEqual([t for t, frame in enumerate(result["frames"]) if frame["keyframe"]], states[0].keyframes)
        self.assertEqual(states[0].keyframes, [1, 2, 3, 4, 5])
        self.assertEqual(counts[:, 0].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(result["max_count"], MAX_COUNT)
        self.assertEqual(result["count_mode"], "event")
        self.assertEqual(result["frames"][0]["prediction"]["fed"], [0, 0])
        for t, frame in enumerate(result["frames"]):
            pred, truth = frame["prediction"], frame["truth"]
            self.assertEqual(pred["change"], [1, 1])
            self.assertEqual(pred["fed"], [t, t])
            self.assertEqual(truth["correction"], [max(-1, min(1, n - t)) for n in truth["count"]])
            previous = result["frames"][t - 1]["truth"]["count"] if t else [0, 0]
            self.assertEqual(truth["change"], [n - p for n, p in zip(truth["count"], previous, strict=True)])
        self.assertNotEqual(result["frames"][-1]["truth"]["change"], result["frames"][-1]["truth"]["correction"])
        last = result["frames"][-1]["prediction"]
        self.assertEqual(len(last["change_probs"][0]), 3)
        self.assertEqual(last["expected"], [float(n) for n in last["count"]])

    def test_evaluate_reports_overall_and_per_task(self):
        model = self.make()
        result = evaluate(model, FakeSet(), model.cfg, 1)
        self.assertEqual(set(result["per_task"]), {"fake"})
        for mode in ("oracle", "model"):
            self.assertTrue(0 <= result["overall"][f"val/{mode}/count_acc"] <= 1)
            self.assertGreater(result["overall"][f"val/{mode}/loss"], 0)
            self.assertIn(result["overall"][f"val/{mode}/count_final_acc"], (0.0, 1.0))
            self.assertIn(f"val/{mode}/count_acc_n1", result["overall"])

    def test_post_press_windows_stop_before_next_press_and_report_misses(self):
        truth = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 1], [2, 1, 1]])
        pred = torch.tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1]])
        events = post_press_events(truth, pred)
        left = [event for event in events if event["item"] == 0]
        self.assertEqual([(e["k"], e["frames"], e["correct"], e["latency"]) for e in left],
                         [(1, 2, 1, 1), (2, 3, 0, None)])
        summary = press_summary(events)
        self.assertEqual(summary["events"], 3)  # 确认键不进入选 best 的分数
        self.assertAlmostEqual(summary["post_press_acc"], 3 / 7)
        self.assertAlmostEqual(summary["miss_rate"], 1 / 3)
        self.assertEqual(summary["latency_median"], 0.5)

    def test_press_validation_uses_all_episodes_fixed_orders_and_restores_mode(self):
        model = self.make(keyframes="none", press_eval=True)
        sample = fake_sample(np.random.default_rng(0), grad_from=0)

        class PressSet:
            episodes = tuple(("press_button", e) for e in range(45, 50))

            def __len__(self):
                return len(self.episodes)

            def sample(self, index, cfg):
                return {**sample, "task": "press_button", "episode": self.episodes[index][1]}

        def perfect(changed, **kwargs):
            return [{"count": (count.to(DEVICE) * 2 - 1) * 20,
                     "stage": torch.zeros(len(changed["steps"]) + 1, device=DEVICE),
                     "progress": torch.zeros(model.cfg.progress_bins, device=DEVICE)} for count in changed["count"]]

        model.train()
        with mock.patch.object(model, "rollout", side_effect=perfect):
            first = press_test(model, PressSet(), model.cfg)
            second = press_test(model, PressSet(), model.cfg)
            combined = evaluate(model, PressSet(), model.cfg, 1)
        self.assertTrue(model.training)
        self.assertEqual(first, second)
        self.assertEqual(len(first["runs"]), 15)
        self.assertEqual(first["overall"]["val/press_irregular/post_press_error"], 0)
        self.assertEqual(combined["episodes"], 1)
        self.assertEqual(len(combined["press_test"]["runs"]), 15)
        for run in first["runs"]:
            self.assertEqual(run["order"][-1], len(sample["frames"]) - 1)
            self.assertEqual(run["truth"], sample["count"][run["order"]].sum(-1).long().tolist())

    def test_pause_test_repeats_a_frame_between_count_changes(self):
        model = self.make()
        result = pause_test(model, FakeSet(), model.cfg, 1, length=3)
        self.assertEqual(result["overall"]["pauses"], 2)      # 计数在第 2、5 步变化：停在第 1、3 步
        self.assertEqual(set(result["per_task"]), {"fake"})
        self.assertTrue(math.isfinite(result["overall"]["count_drift"]))

    def test_pause_export_freezes_images_and_truth_on_expanded_timeline(self):
        from qwen_training.progress.evaluate import PAUSE
        from qwen_training.progress.visualize import export

        sample = fake_sample(np.random.default_rng(0), grad_from=0)
        result = export(self.make(), sample, indices=[0, 2, 4, 6, 8, 10], fps=10, pause=True)
        frames, test = result["frames"], result["pause_test"]
        self.assertEqual(len(frames), len(sample["frames"]) + 2 * PAUSE)
        self.assertEqual(test["overall"]["pauses"], 2)
        self.assertEqual([s["source_step"] for s in test["segments"]], [1, 3])
        self.assertAlmostEqual(frames[-1]["time"], 9.0)
        for segment in test["segments"]:
            start, end = segment["start"], segment["end"]
            self.assertAlmostEqual(frames[end]["time"] - frames[start]["time"], 4.0)
            for frame in frames[start:end + 1]:
                self.assertEqual(frame["images"], frames[start]["images"])
                self.assertEqual(frame["truth"], frames[start]["truth"])
                self.assertEqual(frame["index"], frames[start]["index"])
                self.assertEqual(frame["source_time"], frames[start]["source_time"])
            self.assertEqual(frames[end]["pause"]["offset"], PAUSE)
        self.assertTrue(all(b["time"] > a["time"] for a, b in pairwise(frames)))

    def test_pause_and_export_respect_keyframe_mode(self):
        from qwen_training.progress.visualize import export

        sample = fake_sample(np.random.default_rng(0), grad_from=0)
        for configured, override, expected in (("none", None, "none"), ("label", None, "model"),
                                                ("none", "model", "model"), ("none", "label", "label")):
            with self.subTest(configured=configured, override=override):
                model = self.make(keyframes=configured, delta=0.0)
                with mock.patch.object(model, "rollout", wraps=model.rollout) as rollout:
                    metrics = pause_test(model, FakeSet(), model.cfg, 1, keyframes=override)
                    self.assertEqual(rollout.call_args.kwargs["keyframes"], expected)
                    result = export(model, sample, keyframes=override, pause=True)
                    self.assertEqual(rollout.call_args.kwargs["keyframes"], expected)
                self.assertEqual(metrics["keyframes"], expected)
                self.assertEqual(result["keyframes"], expected)
                self.assertAlmostEqual(metrics["overall"]["count_drift"],
                                       result["pause_test"]["overall"]["count_drift"], places=5)
                if expected == "none":
                    self.assertFalse(any(frame["keyframe"] for frame in result["frames"]))

    def test_eval_and_visualize_cli_preserve_or_override_oracle_feedback(self):
        from qwen_training.progress import evaluate as eval_module
        from qwen_training.progress import visualize
        from qwen_training.progress.model import ProgressModel

        saved = {"keyframes": "none", "count_feedback": True, "oracle_count_feedback": True}
        for module, base in ((eval_module, ["--checkpoint", "unused.pt"]),
                             (visualize, ["--checkpoint", "unused.pt", "--task", "swap_T", "--episode", "49",
                                          "--pause", "--keyframes", "none"])):
            for flags, expected in (([], True), (["--no-oracle-count-feedback"], False)):
                with self.subTest(module=module.__name__, flags=flags), \
                        mock.patch("torch.load", return_value={"config": saved}), \
                        mock.patch.object(ProgressModel, "from_pretrained",
                                          side_effect=RuntimeError("stop before load")) as build:
                    with self.assertRaisesRegex(RuntimeError, "stop before load"):
                        module.main(base + flags)
                    self.assertEqual(build.call_args.args[1].oracle_count_feedback, expected)

    def test_sync_of_item_scores(self):
        t = torch.arange(10.0)
        self.assertAlmostEqual(sync(torch.stack([t, 2 * t + 1, t - 3], 1)), 1.0, places=5)
        self.assertAlmostEqual(sync(torch.stack([t, -t], 1)), -1.0, places=5)
        self.assertIsNone(sync(torch.stack([t, torch.ones(10)], 1)))   # 只有一项在变化


class EventLossTests(unittest.TestCase):
    @staticmethod
    def logits(scores):
        """各项分数 g [T, J] → r = 1…R 的 logit g − (r − 1)。"""
        return torch.tensor(scores, dtype=torch.float)[..., None] - torch.arange(MAX_COUNT)

    @staticmethod
    def labels(counts):
        return (torch.tensor(counts)[..., None] >= torch.arange(1, MAX_COUNT + 1)).float()

    def test_step_is_free_and_time_ramp_is_penalized(self):
        """第 0 项在第 3 步完成，第 1 项没有完成：分数在完成时跳一下、其余不动时损失为 0；同样终点的斜坡要付代价。"""
        counts = [[0, 0]] * 3 + [[1, 0]] * 3
        step = [[0.0, -1.0]] * 3 + [[2.0, -1.0]] * 3
        ramp = [[0.4 * t, -1.0 + 0.1 * t] for t in range(6)]
        self.assertEqual(event_loss(self.logits(step), self.labels(counts)).item(), 0.0)
        self.assertGreater(event_loss(self.logits(ramp), self.labels(counts)).item(), 0.5)
        late = [[0.0, -1.0]] * 5 + [[2.0, -1.0]]     # 晚了两步才跳
        self.assertGreater(event_loss(self.logits(late), self.labels(counts)).item(), 0)

    def test_decrease_needs_a_downward_jump(self):
        counts = [[1]] * 3 + [[0]] * 3
        down, up = [[2.0]] * 3 + [[0.0]] * 3, [[2.0]] * 3 + [[4.0]] * 3
        self.assertEqual(event_loss(self.logits(down), self.labels(counts)).item(), 0.0)
        self.assertGreater(event_loss(self.logits(up), self.labels(counts)).item(), 1.0)

    def test_rare_events_are_not_diluted(self):
        """一次没跳的完成，在 1 项和 20 项不动的项旁边，损失相同（两类各自平均）。"""
        counts = [[0], [0], [0], [1], [1], [1]]
        flat = [[0.0]] * 6
        alone = event_loss(self.logits(flat), self.labels(counts))
        crowd = event_loss(self.logits([row * 21 for row in flat]),
                           self.labels([row + [0] * 20 for row in counts]))
        torch.testing.assert_close(alone, crowd)
        self.assertEqual(alone.item(), 1.0)

    def test_short_segments_have_no_event_loss(self):
        self.assertEqual(event_loss(self.logits([[0.0], [1.0]]), self.labels([[0], [1]])).item(), 0.0)


class ChangeLossTests(unittest.TestCase):
    def outputs(self, logits, fed):
        return [{"stage": torch.zeros(3), "count": logits[t], "progress": torch.zeros(21), "fed": fed[t]}
                for t in range(len(fed))]

    def test_target_corrects_the_fed_count(self):
        """目标 = clip(真值 − 回传, −1, 1)：无误差时是完成事件；回传少 1（漏检）时之后每一步都应 +1，多 1 时应 −1。"""
        sample = fake_sample(np.random.default_rng(0), grad_from=0)          # 第 0 项在第 2 帧、第 1 项在第 5 帧完成
        cfg = tiny_cfg(count_mode="event", progress_bins=21)
        model = mock.Mock(special=torch.zeros(1), bins=torch.linspace(0, 1, 21))
        for fed, plus, minus in ((sample["fed"], {(2, 0), (5, 1)}, set()),
                                 (sample["fed"] - torch.tensor([[0, 0]] * 3 + [[1, 0]] * 3), {(2, 0), (3, 0), (4, 0),
                                                                                              (5, 0), (5, 1)}, set()),
                                 (sample["fed"] + torch.tensor([[0, 0]] * 4 + [[0, 1]] * 2), {(2, 0)}, {(4, 1)})):
            logits = torch.zeros(6, 2, 3, requires_grad=True)
            loss, _ = sample_loss(model, self.outputs(logits, fed), sample, cfg)
            loss.backward()
            push = logits.grad.argmin(-1)                                   # 梯度下降抬高的那一类
            expected = torch.ones(6, 2, dtype=torch.long)
            for t, j in plus:
                expected[t, j] = 2
            for t, j in minus:
                expected[t, j] = 0
            torch.testing.assert_close(push, expected)

    def test_rare_changes_are_not_diluted(self):
        target = torch.ones(100, 1, dtype=torch.long)
        target[50] = 2
        logits = torch.zeros(100, 1, 3, requires_grad=True)
        change_loss(logits, target, torch.ones(100, 1, dtype=torch.bool)).backward()
        self.assertGreater(-logits.grad[50, 0, 2].item(), 10 * -logits.grad[0, 0, 1].item())


class StandardizeTests(unittest.TestCase):
    def test_running_statistics(self):
        from qwen_training.progress.model import Standardize

        norm = Standardize(3, "cpu", momentum=0.5)
        x = torch.tensor([[1.0, 100.0, -5.0], [3.0, 300.0, -5.0]])
        torch.testing.assert_close(norm(x), x)                     # fit 之前原样输出、不更新
        self.assertFalse(norm.initialized)
        norm.fit(x.mean(0), x.pow(2).mean(0))
        torch.testing.assert_close(norm.var, torch.tensor([1.0, 1e4, 0.0]))
        torch.testing.assert_close(norm(x)[:, :2], torch.tensor([[-1.0, -1.0], [1.0, 1.0]]), atol=1e-3, rtol=0)
        norm(x + 10)                                               # 之后按 momentum 滑动一、二阶矩
        torch.testing.assert_close(norm.mean, torch.tensor([7.0, 205.0, 0.0]))
        norm.eval()
        self.assertEqual(norm(torch.tensor([[1e4, 0.0, -5.0]])).abs().max().item(), 5.0)   # 截断到 ±5
        torch.testing.assert_close(norm.mean, torch.tensor([7.0, 205.0, 0.0]))   # 验证时不更新

    def test_variance_spans_calls(self):
        """每次调用内各行相同（如复制到 J 项的 h_CNT），调用间不同：方差来自调用之间，不能为 0。"""
        from qwen_training.progress.model import Standardize

        norm = Standardize(1, "cpu", momentum=0.5)
        norm.fit(torch.zeros(1), torch.ones(1))
        for value in (1.0, -1.0, 1.0, -1.0):
            norm(torch.full((3, 1), value))
        self.assertGreater(norm.var.item(), 0.5)


class SamplingTests(unittest.TestCase):
    def test_resample_keeps_labels_with_source_frames(self):
        frames = [{"index": 4 * t, "stage": t // 3, "counts": [t // 3], "progress": t / 9} for t in range(10)]
        slow = data.resample(frames, 0.5)
        self.assertEqual(len(slow), 19)
        self.assertEqual([f["index"] for f in slow], list(range(0, 37, 2)))        # 真实中间帧，不重复
        self.assertAlmostEqual(slow[1]["progress"], 0.5 / 9)
        self.assertEqual((slow[5]["stage"], slow[5]["counts"]), (0, [0]))          # 位置 2.5 还在第 0 段
        fast = data.resample(frames, 2.0)
        self.assertEqual([f["index"] for f in fast], [0, 8, 16, 24, 32, 36])       # 末帧总在最后
        self.assertEqual(fast[-1]["progress"], 1)
        self.assertEqual(data.resample(frames, 1.0), frames)
        self.assertEqual(data.resample(frames[:1], 2.0), frames[:1])

    def test_warp_keeps_labels_with_frames_and_ends_on_last_frame(self):
        cfg = tiny_cfg(pause_prob=1.0, speed_prob=1.0)
        for seed in range(50):
            order = data.warp(30, cfg, random.Random(seed))
            self.assertEqual(order[-1], 29)
            self.assertEqual(order, sorted(order))              # 只重复、删帧，不打乱顺序
            self.assertEqual(order[0], 0)
        self.assertEqual(data.warp(30, tiny_cfg(), random.Random(0)), list(range(30)))
        lengths = [len(data.warp(30, tiny_cfg(pause_prob=1.0), random.Random(seed))) for seed in range(20)]
        self.assertTrue(all(35 <= n <= 90 for n in lengths))

    def test_warp_rewinds_then_replays_forward(self):
        cfg = tiny_cfg(rewind_prob=1.0)
        for seed in range(50):
            order = data.warp(30, cfg, random.Random(seed), rewind_safe=True)
            self.assertEqual((order[0], order[-1]), (0, 29))
            back = [i for i in range(1, len(order)) if order[i] < order[i - 1]]
            self.assertTrue(1 <= len(back) <= 20)
            self.assertEqual(back, list(range(back[0], back[-1] + 1)))     # 只倒放一段，连续
            self.assertEqual(len(order), 30 + 2 * len(back))               # 倒回去几帧，就重做几帧
            replay = order[back[-1]:]
            self.assertEqual(replay, list(range(replay[0], 30)))            # 之后从回退点正放到末帧

    def test_rewind_requires_task_approval_and_nonzero_probability(self):
        self.assertEqual((Config().pause_prob, Config().speed_prob, Config().rewind_prob), (0.2, 0.2, 0.0))
        self.assertTrue(all(not spec.rewind_safe for spec in TASKS.values()))
        for seed in range(20):
            self.assertEqual(data.warp(30, tiny_cfg(rewind_prob=1.0), random.Random(seed)), list(range(30)))
            self.assertEqual(data.warp(30, tiny_cfg(rewind_prob=0.0), random.Random(seed), rewind_safe=True),
                             list(range(30)))

    def test_warp_keeps_the_anchor_frame(self):
        cfg = tiny_cfg(speed_prob=1.0)
        self.assertNotIn(3, data.warp(30, cfg, random.Random(6)))           # 不保留时 2 倍速会删掉它
        for seed in range(100):
            self.assertIn(3, data.warp(30, cfg, random.Random(seed), keep=3))

    def test_repeat_boost_balances_tasks(self):
        dataset = mock.Mock(episodes=[("press_button", 9), ("press_button", 10)] + [("swap_T", e) for e in range(4)])
        stream = TrainStream(dataset, tiny_cfg(repeat_boost=2.0), 0)
        press, swap = sum(stream.weights[:2]), sum(stream.weights[2:])
        self.assertAlmostEqual(press / swap, 2.0)
        self.assertIsNone(TrainStream(dataset, tiny_cfg(), 0).weights)

    def test_epoch_sampling_covers_all_episodes_and_balances_tasks(self):
        dataset = mock.Mock(episodes=[("press_button", 9), ("press_button", 10)] +
                           [("swap_T", e) for e in range(4)])
        a = EpochSamples(dataset, tiny_cfg(repeat_boost=2), 0)
        self.assertEqual(Counter(a.order), {0: 4, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1})
        self.assertEqual(a.order, EpochSamples(dataset, tiny_cfg(repeat_boost=2), 0).order)
        self.assertNotEqual(a.order, EpochSamples(dataset, tiny_cfg(repeat_boost=2), 1).order)
        self.assertEqual(sorted(EpochSamples(dataset, tiny_cfg(), 0).order), list(range(6)))


class ConfigTests(unittest.TestCase):
    def test_stage_configs_allow_independent_segment_overrides(self):
        configs = Path(__file__).resolve().parents[1] / "qwen_training/progress/configs"
        for stage, teacher, segment in ((1, True, 9), (2, False, 3)):
            args = ["--config", str(configs / f"rmbench_2b_stage{stage}.yaml"), "--segment", str(segment)]
            if stage == 2:
                args += ["--init-from", "stage1.pt"]
            cfg = parse_config(args)
            self.assertEqual((cfg.teacher, cfg.segment), (teacher, segment))
            if stage == 2:
                self.assertEqual(cfg.init_from, "stage1.pt")

    def test_yaml_then_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            path.write_text("segment: 8\nlr: 2.0e-4\n")
            cfg = parse_config(["--config", str(path), "--segment", "4", "--isolate-gdn", "false"])
            self.assertEqual((cfg.segment, cfg.lr, cfg.isolate_gdn), (4, 2e-4, False))
            path.write_text("segmnet: 8\n")
            with self.assertRaises(ValueError):
                parse_config(["--config", str(path)])

    def test_checkpoint_config_keeps_old_feedback_and_applies_overrides(self):
        from qwen_training.progress.config import checkpoint_config

        self.assertFalse(Config().count_feedback)
        self.assertEqual(Config().keyframes, "none")                         # 不保留历史关键帧；首帧和局部上下文仍保留
        self.assertFalse(Config().count_reads_history)
        self.assertFalse(Config().progress_reads_history)
        self.assertTrue(checkpoint_config({}).progress_reads_history)
        self.assertEqual(checkpoint_config({}).keyframes, "label")
        self.assertEqual(checkpoint_config({"keyframes": "none"}).keyframes, "none")
        self.assertFalse(checkpoint_config({}).stage_reads_history)
        renamed = checkpoint_config({"p_reads_count": False, "p_reads_history": False})   # 旧名按新名读
        self.assertEqual((renamed.progress_reads_count, renamed.progress_reads_history), (False, False))
        self.assertFalse(checkpoint_config({"count_reads_history": False}).count_reads_history)
        self.assertTrue(checkpoint_config({}).count_feedback)                  # 旧 checkpoint 没有这一项
        self.assertFalse(checkpoint_config({"count_feedback": False}).count_feedback)
        cfg = checkpoint_config({}, count_feedback=None, progress_reads_count=False, count_reads_history=None)
        self.assertEqual((cfg.count_feedback, cfg.progress_reads_count, cfg.count_reads_history), (True, False, True))
        with self.assertRaises(ValueError):
            checkpoint_config({"count_feedback": False}, oracle_count_feedback=True)


PUT_BACK = [["Pick up the block and move it to the center position.", 10], ["Press the button.", 10],
            ["Move the block back to the front mat.", 10]]
PRESS = [["Press the left button one time.", 10], ["Press the left button one time.", 10],
         ["Press the middle button one time.", 10], ["Press the confirm button.", 10]]
BATTERY = [["Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.",
            10], ["Pick up the left battery and place it into the battery slot in the negative direction.", 10]]


class DataTests(unittest.TestCase):
    """用 labels.py 生成 50 条标签，HDF5 读取换成假的，检查划分、示范来源和样本里的标签。"""

    def sample(self, task, segments, split="val", segment=None, rng=None, anchor=None, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / task
            labels.mkdir()
            for episode in range(50):
                record = {"task": task, "episode": episode, "kind": TASKS[task].kind, "instruction": "do it",
                          "source": f"{task}/episode{episode}.hdf5", **episode_labels(segments, TASKS[task], 25)}
                (labels / f"episode{episode}.json").write_text(json.dumps(record))
            dataset = data.RMBench(tmp, tmp, split)
            self.levels = dataset.count_levels()
            self.reads = []

            def read(path, indices, cameras=data.CAMERAS):
                self.reads.append(Path(path).name)
                return [[np.zeros((4, 4, 3), np.uint8)] * len(cameras) for _ in indices]

            with mock.patch.object(data, "read_frames", read):
                return dataset, dataset.sample(0, tiny_cfg(boundary=1, **overrides), rng, segment, anchor)

    def test_validation_sample_labels(self):
        dataset, sample = self.sample("put_back_block", PUT_BACK)
        self.assertEqual([episode for _, episode in dataset.episodes], list(range(45, 50)))
        self.assertEqual(self.reads, ["episode8.hdf5", "episode45.hdf5"])    # 验证用固定评估示范
        # 5 Hz 采样的原始帧号 0, 5, …, 25, 29；三个实例在第 9、19、29 帧结束。
        self.assertEqual(sample["keyframes"].nonzero().flatten().tolist(), [2, 4, 6])
        self.assertFalse(sample["extra_keyframes"].any())                      # 验证不随机加帧
        self.assertEqual(sample["stage_mask"].tolist(), [True, False, False, False, False, True, True])
        self.assertEqual(sample["count"].shape, (7, 3, MAX_COUNT))
        self.assertEqual(sample["count"][-1].sum(-1).tolist(), [1, 1, 1])

    def test_segment_end_keeps_boundary_masks_from_full_episode(self):
        _, full = self.sample("put_back_block", PUT_BACK)
        for segment, seed in ((3, 0), (4, 2)):
            _, short = self.sample("put_back_block", PUT_BACK, "train", segment, random.Random(seed))
            end = len(short["frames"])
            for name in ("stage_mask", "count_mask"):
                torch.testing.assert_close(short[name], full[name][:end])

    def test_terminal_count_target_is_kept(self):
        """最后一项在末帧（第 6 步）才完成，那是它唯一的正标签：末帧不屏蔽，前一帧照常屏蔽。"""
        _, sample = self.sample("put_back_block", PUT_BACK)
        self.assertEqual(sample["count"][-1, 2, 0].item(), 1.0)
        self.assertTrue(sample["count_mask"][-1].all())
        self.assertFalse(sample["count_mask"][-2, 2, 0])
        self.assertFalse(sample["count_mask"][3, 1, 0])           # 中间的完成（第 4 步）前后照常屏蔽
        # 带到末帧的训练段同样保留；在中途截断的段，段尾不因"是段的最后一帧"而解除屏蔽。
        _, tail = self.sample("put_back_block", PUT_BACK, "train", 3, random.Random(5))   # 起点 4，段到末帧
        self.assertEqual(len(tail["frames"]), 7)
        self.assertTrue(tail["count_mask"][-1].all())
        _, cut = self.sample("put_back_block", PUT_BACK, "train", 3, random.Random(0))
        self.assertLess(len(cut["frames"]), 7)
        self.assertFalse(cut["count_mask"][-1].all())

    def test_terminal_count_loss_raises_the_score(self):
        """末帧上最后一项的分数 g = 0 时，梯度下降应把 g 往上推（修复前 r ≥ 2 的负标签会把它压低）。"""
        _, sample = self.sample("put_back_block", PUT_BACK)
        steps, bins = len(sample["frames"]), torch.linspace(0, 1, 21)
        g = torch.zeros(3, requires_grad=True)
        outputs = [{"stage": torch.zeros(4), "count": g[:, None] - torch.arange(MAX_COUNT) if t == steps - 1
                    else torch.full((3, MAX_COUNT), -5.0), "progress": torch.zeros(21)} for t in range(steps)]
        model = mock.Mock(special=torch.zeros(1), bins=bins)
        loss, _ = sample_loss(model, outputs, sample, tiny_cfg(progress_bins=21))
        loss.backward()
        self.assertLess(g.grad[2].item(), 0)

    def test_event_segment_contains_a_count_change(self):
        """计数在第 2、4、6 步变化（6 为末帧）；event_prob=1、段长 1 时，带梯度的那一步必须是其中之一。"""
        for seed in range(20):
            _, sample = self.sample("put_back_block", PUT_BACK, "train", 1, random.Random(seed), event_prob=1.0)
            self.assertIn(sample["grad_from"], (2, 4, 6))

    def test_count_levels_and_anchor(self):
        """press_button 左键 2 次：各项最大计数依次经过 0、1、2；段包含 anchor 帧。"""
        _, full = self.sample("press_button", PRESS)
        top = full["count"].sum(-1).amax(-1).tolist()                 # 验证样本 = episode 45 的整条
        self.assertEqual(len(self.levels), data.COUNT_LEVELS)
        self.assertEqual(sorted({t for index, t in self.levels[2] if index == 0}),
                         [t for t, c in enumerate(top) if c == 2])
        self.assertEqual(sum(map(len, self.levels)), 5 * len(top))
        for seed in range(20):
            _, sample = self.sample("press_button", PRESS, "train", 2, random.Random(seed), anchor=3)
            self.assertIn(sample["grad_from"], (2, 3))
            self.sample("press_button", PRESS, "train", 2, random.Random(seed), anchor=3, speed_prob=1.0)   # 不抛错

    def test_count_balance_draws_levels_uniformly(self):
        levels = [[(0, 0)] * 90, [(1, 5)] * 9, [(2, 7)], [], []]
        dataset = mock.Mock(episodes=[("press_button", 9)] * 3, count_levels=lambda: levels)
        dataset.sample.side_effect = lambda index, cfg, rng, segment, anchor=None: index
        stream = iter(TrainStream(dataset, tiny_cfg(count_balance=True), 0))
        drawn = Counter(next(stream) for _ in range(3000))
        for index in range(3):
            self.assertAlmostEqual(drawn[index] / 3000, 1 / 3, delta=0.04)

    def test_fed_counts_and_injected_errors(self):
        """put_back_block 三项依次在第 2、4、6 步完成。fed = 上一帧的真值；注入的漏检少 1、误检多 1，持续到末帧。"""
        _, clean = self.sample("put_back_block", PUT_BACK)
        truth = clean["count"].sum(-1).long()
        torch.testing.assert_close(clean["fed"][1:], truth[:-1])
        self.assertEqual(clean["fed"][0].tolist(), [0, 0, 0])
        # 完成帧计损失，前后 ±1 步不计
        self.assertEqual(clean["event_mask"][:, 1].tolist(), [True, True, True, False, True, False, True])
        seen = set()
        for seed in range(40):
            _, sample = self.sample("put_back_block", PUT_BACK, "train", None, random.Random(seed), inject_prob=1.0)
            diff = sample["fed"] - torch.cat([torch.zeros_like(truth[:1]), truth[:-1]])
            changed = diff.nonzero().tolist()
            self.assertTrue(changed)
            items = {j for _, j in changed}
            self.assertEqual(len(items), 1)                                  # 只有一项出错
            (j,) = items
            first = min(t for t, _ in changed)
            self.assertTrue((diff[first:, j] == diff[first, j]).all())        # 一直持续到末帧
            self.assertIn(diff[first, j].item(), (-1, 1))
            if diff[first, j] == -1:                                         # 漏检：恰好在那次完成之后开始
                self.assertEqual(first, 2 * j + 3)
            seen.add(diff[first, j].item())
        self.assertEqual(seen, {-1, 1})

    def test_warped_sample_labels_follow_frames(self):
        _, full = self.sample("put_back_block", PUT_BACK)
        _, warped = self.sample("put_back_block", PUT_BACK, "train", None, random.Random(3), pause_prob=1.0,
                                speed_prob=1.0, keyframe_drop=0.0, keyframe_jitter=0)
        self.assertGreater(len(warped["frames"]), 7)
        rows = {(s, tuple(c.flatten().tolist()), p) for s, c, p in
                zip(full["stage"].tolist(), full["count"], full["progress"].tolist(), strict=True)}
        for s, c, p in zip(warped["stage"].tolist(), warped["count"], warped["progress"].tolist(), strict=True):
            self.assertIn((s, tuple(c.flatten().tolist()), p), rows)
        self.assertEqual(warped["count"][-1].sum(-1).tolist(), [1, 1, 1])
        self.assertTrue(warped["count_mask"][-1].all())
        self.assertEqual(warped["keyframes"].sum().item(), 3)   # 每次完成仍各有一个关键帧

    def test_unapproved_tasks_never_rewind_even_when_requested(self):
        for task, segments in (("press_button", PRESS), ("put_back_block", PUT_BACK)):
            for seed in range(10):
                _, sample = self.sample(task, segments, "train", None, random.Random(seed), rewind_prob=1.0)
                self.assertTrue((sample["progress"][1:] >= sample["progress"][:-1]).all())
                self.assertTrue((sample["count"][1:] >= sample["count"][:-1]).all())

    def test_rewound_sample_progress_and_count_go_back(self):
        from dataclasses import replace

        # 只在测试中临时批准，用于验证倒放机制；真实任务保持未批准。
        with mock.patch.dict(TASKS, put_back_block=replace(TASKS["put_back_block"], rewind_safe=True)):
            rewound = [self.sample("put_back_block", PUT_BACK, "train", None, random.Random(seed), rewind_prob=1.0)[1]
                       for seed in range(20)]
        for sample in rewound:
            self.assertEqual(sample["count"][-1].sum(-1).tolist(), [1, 1, 1])   # 重做之后末帧计数仍是真值
        self.assertTrue(any((s["progress"][1:] < s["progress"][:-1]).any() for s in rewound))
        self.assertTrue(any((s["count"].sum((-2, -1))[1:] < s["count"].sum((-2, -1))[:-1]).any() for s in rewound))

    def test_count_margin_ignores_far_thresholds(self):
        _, sample = self.sample("put_back_block", PUT_BACK)
        steps = len(sample["frames"])
        count = torch.zeros(steps, 3, MAX_COUNT, requires_grad=True)
        outputs = [{"stage": torch.zeros(4), "count": count[t], "progress": torch.zeros(21)} for t in range(steps)]
        model = mock.Mock(special=torch.zeros(1), bins=torch.linspace(0, 1, 21))
        loss, _ = sample_loss(model, outputs, sample, tiny_cfg(progress_bins=21), margin=2)
        loss.backward()
        # 末帧各项计数为 1：r ≤ 3 的门槛有梯度，r ≥ 4 没有；第 0 帧计数为 0：只有 r ≤ 2。
        self.assertTrue((count.grad[-1, :, :3] != 0).all())
        self.assertTrue((count.grad[-1, :, 3:] == 0).all())
        self.assertTrue((count.grad[0, :, 2:] == 0).all() and (count.grad[0, :, :2] != 0).all())

    def test_train_sample_draws_demo_from_pool(self):
        dataset, sample = self.sample("put_back_block", PUT_BACK, "train", segment=3, rng=random.Random(0))
        self.assertEqual([episode for _, episode in dataset.episodes], list(range(9, 45)))
        self.assertIn(self.reads[0], {f"episode{n}.hdf5" for n in range(8)})
        self.assertEqual(len(sample["frames"]), sample["grad_from"] + 3)

    def test_counts_for_repeated_and_search_tasks(self):
        _, press = self.sample("press_button", PRESS)          # 左 2、中 1、确认 1
        self.assertEqual(press["count"].shape[1:], (3, MAX_COUNT))
        self.assertEqual(press["count"][-1].sum(-1).tolist(), [2, 1, 1])
        self.assertEqual(press["count"][-1, 0].tolist(), [1, 1] + [0] * (MAX_COUNT - 2))
        _, battery = self.sample("battery_try", BATTERY)       # 首次放入 1 次、翻转 1 次
        self.assertEqual(battery["count"][-1].sum(-1).tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
