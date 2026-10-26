# ruff: noqa: RUF002, RUF003
"""用随机初始化的小 Qwen3.5 检查流式前向、可见性隔离、反传与数据样本。有 GPU 时在 GPU 上跑（走 fla kernel）。"""

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from qwen_training.datasets.rmbench.labels import MAX_COUNT, episode_labels
from qwen_training.datasets.rmbench.tasks import TASKS
from qwen_training.progress import data
from qwen_training.progress.config import Config, parse_config
from qwen_training.progress.evaluate import evaluate
from qwen_training.progress.loss import autocast, sample_loss
from qwen_training.progress.stream import CNT, FRAME, PREFIX, STG, P, StreamState, Tokens, forward_chunk, visibility
from qwen_training.progress.train import train

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
    values = dict(head_hidden=32, lora_rank=4, window=3, max_keyframes=2, segment=4, device=DEVICE, accumulate=1,
                  iterations=2, log_every=1, eval_every=2, eval_episodes=1, save_every=100, workers=0)
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
        torch.testing.assert_close(other[:6], base[:6])          # 第二步画面和 [STG] 不变
        self.assertFalse(torch.allclose(other[6], base[6]))      # [CNT] 接力看到上一步 [CNT]
        self.assertFalse(torch.allclose(other[7], base[7]))
        leaky, leaky_other = self.run_two_steps(readouts, False), self.run_two_steps(changed, False)
        self.assertFalse(torch.allclose(leaky[:5], leaky_other[:5]))   # 不隔离时经 GDN 漏给画面

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
                    [1, 1, 1, 0, 1, 0, 1],    # [CNT]：加关键帧和之前的 [CNT]
                    [1, 1, 1, 1, 1, 1, 1]]    # [P]：全部
        self.assertEqual(mask.int().tolist(), expected)

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
        model = self.make(window=2, keyframes="none")
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

    def test_evaluate_without_keyframes_keeps_default_best_metric(self):
        model = self.make(keyframes="none")
        result = evaluate(model, FakeSet(), model.cfg, 1)
        self.assertIn(model.cfg.best_metric, result["overall"])
        self.assertFalse(any(key.startswith("val/oracle/") for key in result["overall"]))

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

    def test_evaluate_reports_overall_and_per_task(self):
        model = self.make()
        result = evaluate(model, FakeSet(), model.cfg, 1)
        self.assertEqual(set(result["per_task"]), {"fake"})
        for mode in ("oracle", "model"):
            self.assertTrue(0 <= result["overall"][f"val/{mode}/count_acc"] <= 1)
            self.assertGreater(result["overall"][f"val/{mode}/loss"], 0)


class ConfigTests(unittest.TestCase):
    def test_yaml_then_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.yaml"
            path.write_text("segment: 8\nlr: 2.0e-4\n")
            cfg = parse_config(["--config", str(path), "--segment", "4", "--isolate-gdn", "false"])
            self.assertEqual((cfg.segment, cfg.lr, cfg.isolate_gdn), (4, 2e-4, False))
            path.write_text("segmnet: 8\n")
            with self.assertRaises(ValueError):
                parse_config(["--config", str(path)])


PUT_BACK = [["Pick up the block and move it to the center position.", 10], ["Press the button.", 10],
            ["Move the block back to the front mat.", 10]]
PRESS = [["Press the left button one time.", 10], ["Press the left button one time.", 10],
         ["Press the middle button one time.", 10], ["Press the confirm button.", 10]]
BATTERY = [["Use dual arm to pick up the batteries and place them into the battery slots in the positive direction.",
            10], ["Pick up the left battery and place it into the battery slot in the negative direction.", 10]]


class DataTests(unittest.TestCase):
    """用 labels.py 生成 50 条标签，HDF5 读取换成假的，检查划分、示范来源和样本里的标签。"""

    def sample(self, task, segments, split="val", segment=None, rng=None):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / task
            labels.mkdir()
            for episode in range(50):
                record = {"task": task, "episode": episode, "kind": TASKS[task].kind, "instruction": "do it",
                          "source": f"{task}/episode{episode}.hdf5", **episode_labels(segments, TASKS[task], 25)}
                (labels / f"episode{episode}.json").write_text(json.dumps(record))
            dataset = data.RMBench(tmp, tmp, split)
            self.reads = []

            def read(path, indices, cameras=data.CAMERAS):
                self.reads.append(Path(path).name)
                return [[np.zeros((4, 4, 3), np.uint8)] * len(cameras) for _ in indices]

            with mock.patch.object(data, "read_frames", read):
                return dataset, dataset.sample(0, tiny_cfg(boundary=1), rng, segment)

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
