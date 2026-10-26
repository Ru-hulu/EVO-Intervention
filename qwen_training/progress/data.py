# ruff: noqa: RUF002, RUF003
"""RMBench 训练样本：示范池抽示范，当前 episode 取热身 + 带梯度的一段（proposal 3.2.1、5.1、5.2）。

每个任务 50 条 episode 的默认划分：0–7 示范池，8 评估示范，9–44 训练，45–49 验证。
示范池和评估示范的 episode 不作为当前 episode。
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

from qwen_training.datasets.rmbench.labels import CAMERAS, MAX_COUNT
from qwen_training.datasets.rmbench.tasks import TASKS

SPLIT = {"pool": range(0, 8), "eval_demo": 8, "train": range(9, 45), "val": range(45, 50)}


def read_frames(path, indices, cameras=CAMERAS):
    """[[每个视角的 RGB 数组] for 每个帧号]。

    RMBench 旧格式 JPEG 用 cv2 解出来正好是 RGB；带 XPL-RGB1 标记的是标准 JPEG，要从 BGR 转。
    """
    import cv2
    import h5py

    frames = []
    with h5py.File(path, "r") as f:
        for index in indices:
            views = []
            for camera in cameras:
                buffer = np.frombuffer(f[f"observation/{camera}/rgb"][index], np.uint8)
                image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                views.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if b"XPL-RGB1" in buffer.tobytes() else image)
            frames.append(views)
    return frames


class RMBench:
    def __init__(self, labels, data, split, tasks=None):
        self.labels, self.data = Path(labels), Path(data)
        self.episodes = [(task, episode) for task in sorted(tasks or TASKS)
                         for episode in SPLIT[split]
                         if (self.labels / task / f"episode{episode}.json").exists()]
        if not self.episodes:
            raise ValueError(f"{self.labels} 下没有 {split} 划分的标签")

    def __len__(self):
        return len(self.episodes)

    def sample(self, index, cfg, rng=None, segment=None):
        """rng 为 None 时是验证样本：整条 episode、评估示范、不做增强。segment 为带梯度的步数。"""
        train = rng is not None
        rng = rng or random.Random(index)
        task, episode = self.episodes[index]
        record = json.loads((self.labels / task / f"episode{episode}.json").read_text())
        frames = record["frames"]
        grad_from, end = 0, len(frames)
        if segment is not None:
            grad_from = rng.randint(0, max(0, len(frames) - segment))
            end = min(len(frames), grad_from + segment)

        # 累积计数标签 [T, J, R]：第 j 项是否已经做了至少 r 次，r = 1…MAX_COUNT；三类任务相同。
        stage = torch.tensor([frame["stage"] for frame in frames])
        count = (torch.tensor([frame["counts"] for frame in frames])[..., None]
                 >= torch.arange(1, MAX_COUNT + 1)).float()

        # 切换、完成前后 ±boundary 步不计 STG、CNT 损失（CNT 按每个 (j, r) 各自算）。
        masks = {}
        for name, values in (("stage", stage), ("count", count)):
            changed = torch.zeros(values.shape, dtype=torch.bool)
            for shift in range(1, cfg.boundary + 1):
                diff = values[shift:] != values[:-shift]
                changed[shift:] |= diff
                changed[:-shift] |= diff
            masks[f"{name}_mask"] = ~changed[:end]

        frames, stage, count = frames[:end], stage[:end], count[:end]

        # 标注关键帧 = 任一项计数变化的帧（子任务边界）；训练时随机偏移、丢弃（proposal 5.1）。
        # 训练时另外随机加入无关帧，模拟推理时的误触发；单独记录，关键帧总数超上限时先删它们。
        keyframes = torch.zeros(len(frames), dtype=torch.bool)
        jitter, drop, extra = (cfg.keyframe_jitter, cfg.keyframe_drop, cfg.keyframe_extra) if train else (0, 0, 0)
        for t in ((count[1:] != count[:-1]).flatten(1).any(-1).nonzero().flatten() + 1).tolist():
            if rng.random() >= drop:
                keyframes[min(max(t + rng.randint(-jitter, jitter), 0), len(frames) - 1)] = True
        extra_keyframes = torch.tensor([rng.random() < extra for _ in frames], dtype=torch.bool) & ~keyframes

        # 示范：训练从示范池随机抽一条，关键帧时刻扰动、随机只留头部视角；验证用固定评估示范。
        demo_episode = rng.choice(list(SPLIT["pool"])) if train else SPLIT["eval_demo"]
        demo = json.loads((self.labels / task / f"episode{demo_episode}.json").read_text())
        jitter = cfg.demo_jitter if train else 0
        demo_frames = [min(max(item["frame"] + rng.randint(-jitter, jitter), 0), demo["trace"][-1]["end"])
                       for item in demo["demo"]]
        cameras = CAMERAS[:1] if train and rng.random() < cfg.drop_wrist else CAMERAS
        demo_views = read_frames(self.data / demo["source"], demo_frames, cameras)

        return {"task": task, "episode": episode, "instruction": record["instruction"], "steps": record["steps"],
                "demo": [{"step": item["step"], "views": views}
                         for item, views in zip(demo["demo"], demo_views, strict=True)],
                "frames": read_frames(self.data / record["source"], [frame["index"] for frame in frames]),
                "grad_from": grad_from, "stage": stage, "count": count,
                "progress": torch.tensor([frame["progress"] for frame in frames]), "total": len(record["trace"]),
                "keyframes": keyframes, "extra_keyframes": extra_keyframes, **masks}
