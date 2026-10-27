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
COUNT_LEVELS = 5   # count_balance 的等级：各项最大计数 0、1、2、3、≥4


def read_video(path, indices):
    """按原视频帧号读 RGB；保留请求顺序和时间增强产生的重复帧。"""
    import av

    if not indices:
        return []
    wanted, decoded = set(indices), {}
    if min(wanted) < 0:
        raise ValueError("视频帧号必须非负")
    last = max(wanted)
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                decoded[index] = frame.to_ndarray(format="rgb24")
            if index >= last:
                break
    if wanted - decoded.keys():
        raise ValueError(f"{path}: 无法读取帧 {sorted(wanted - decoded.keys())}")
    return [decoded[index] for index in indices]


def read_frames(path, indices, cameras=CAMERAS):
    """[[每个视角的 RGB 数组] for 每个帧号]。

    RMBench 旧格式 JPEG 用 cv2 解出来正好是 RGB；带 XPL-RGB1 标记的是标准 JPEG，要从 BGR 转。
    """
    if isinstance(path, dict):
        views = [read_video(path[camera], indices) for camera in cameras]
        return [[view[t] for view in views] for t in range(len(indices))]
    if Path(path).suffix.lower() == ".mp4":
        return [[frame] for frame in read_video(path, indices)]

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


def warp(n, cfg, rng, keep=None):
    """时间增强：返回新的帧下标序列，标签跟着帧走，末帧总在最后。

    - 变速：以 speed_prob 随机选一段，隔一帧删一帧（2 倍速）或每帧重复一次（0.5 倍速）；2 倍速不删 keep 这一帧；
    - 停顿：以 pause_prob 在 1–3 处把某一帧重复 5–20 步，期间 stage、计数、p 都不变。
    """
    order = list(range(n))
    if n > 2 and rng.random() < cfg.speed_prob:
        a, b = sorted(rng.sample(range(n), 2))
        part = [i for k, i in enumerate(order[a:b]) if k % 2 == 0 or i == keep] if rng.random() < 0.5 \
            else [i for i in order[a:b] for _ in range(2)]
        order = order[:a] + part + order[b:]
    if rng.random() < cfg.pause_prob:
        for _ in range(rng.randint(1, 3)):
            t = rng.randrange(len(order))
            order[t:t] = [order[t]] * rng.randint(5, 20)
    return order


class RMBench:
    def __init__(self, labels, data, split, tasks=None):
        self.labels, self.data = Path(labels), Path(data)
        manifest = self.labels / "_splits.json"
        self.splits = json.loads(manifest.read_text()) if manifest.exists() else {task: SPLIT for task in TASKS}
        self.video_sources = None
        index = self.data / "metadata.jsonl"
        if index.exists() and not manifest.exists():
            rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
            self.video_sources = {
                (row["task_slug"], row["episode_index"]): {
                    view["camera"]: self.data / view["video_path"] for view in row["video_views"].values()}
                for row in rows if row["task_slug"] in TASKS}
        self.episodes = [(task, episode) for task in sorted(tasks or self.splits)
                         for episode in self.splits[task][split]
                         if (self.labels / task / f"episode{episode}.json").exists()
                         and (self.video_sources is None or (task, episode) in self.video_sources)]
        if not self.episodes:
            raise ValueError(f"{self.labels} 下没有 {split} 划分的标签")

    def __len__(self):
        return len(self.episodes)

    def source(self, record):
        if self.video_sources is None:
            return self.data / record["source"]
        return self.video_sources[record["task"], record["episode"]]

    def count_levels(self):
        """按计数等级给所有帧分桶：第 c 个桶是"各项最大计数 = c"的 (episode 下标, 帧号)，最后一桶为 ≥ COUNT_LEVELS−1。"""
        levels = [[] for _ in range(COUNT_LEVELS)]
        for index, (task, episode) in enumerate(self.episodes):
            frames = json.loads((self.labels / task / f"episode{episode}.json").read_text())["frames"]
            for t, frame in enumerate(frames):
                levels[min(max(frame["counts"]), COUNT_LEVELS - 1)].append((index, t))
        return levels

    def sample(self, index, cfg, rng=None, segment=None, anchor=None):
        """rng 为 None 时是验证样本：整条 episode、评估示范、不做增强。segment 为带梯度的步数。

        anchor：带梯度的段必须包含的帧号（原始帧序，时间增强后取它第一次出现的位置）。
        """
        train = rng is not None
        rng = rng or random.Random(index)
        task, episode = self.episodes[index]
        record = json.loads((self.labels / task / f"episode{episode}.json").read_text())
        frames = record["frames"]
        if train and (cfg.pause_prob or cfg.speed_prob):
            order = warp(len(frames), cfg, rng, anchor)
            frames = [frames[i] for i in order]
            anchor = None if anchor is None else order.index(anchor)

        # 累积计数标签 [T, J, R]：第 j 项是否已经做了至少 r 次，r = 1…MAX_COUNT；三类任务相同。
        stage = torch.tensor([frame["stage"] for frame in frames])
        count = (torch.tensor([frame["counts"] for frame in frames])[..., None]
                 >= torch.arange(1, MAX_COUNT + 1)).float()
        changes = ((count[1:] != count[:-1]).flatten(1).any(-1).nonzero().flatten() + 1).tolist()

        # 带梯度的段：均匀抽起点；给了 anchor 时让段包含它，否则以 event_prob 先抽一个计数变化帧（含末帧）作 anchor。
        grad_from, end = 0, len(frames)
        if segment is not None:
            if anchor is None and cfg.event_prob and rng.random() < cfg.event_prob:
                anchor = rng.choice([*changes, len(frames) - 1])
            if anchor is not None:
                grad_from = rng.randint(max(0, anchor - segment + 1), max(0, min(anchor, len(frames) - segment)))
            else:
                grad_from = rng.randint(0, max(0, len(frames) - segment))
            end = min(len(frames), grad_from + segment)

        # 切换、完成前后 ±boundary 步不计 STG、CNT 损失（CNT 按每个 (j, r) 各自算）。
        # 整条 episode 的末帧除外：最后一项恰好在末帧完成，那是它唯一的正标签；此时任务已成功，计数是确定的。
        masks = {}
        for name, values in (("stage", stage), ("count", count)):
            changed = torch.zeros(values.shape, dtype=torch.bool)
            for shift in range(1, cfg.boundary + 1):
                diff = values[shift:] != values[:-shift]
                changed[shift:] |= diff
                changed[:-shift] |= diff
            if name == "count":
                changed[-1] = False
            masks[f"{name}_mask"] = ~changed[:end]

        # 事件模式（count_mode=event）：fed[t, j] 是第 t 步回传给 [CNT] 的第 j 项次数（第 t−1 帧之后的状态），
        # 无注入时为真值 n(t−1)。训练时以 inject_prob 注入一处推理时会出现的错误，持续到末帧：
        # 漏检——某次真实完成没加上，之后少 1；误检——某一帧凭空加 1，之后多 1。回传序列里没有"突然掉 1"的痕迹，
        # 模型只能对照画面发现。事件目标 = clip(n(t) − fed(t), −1, 1)（见 loss.sample_loss），出错后就是纠正。
        n = count.sum(-1).long()
        fed = torch.cat([torch.zeros_like(n[:1]), n[:-1]])
        if train and rng.random() < cfg.inject_prob:
            # (c − 1, j)：第 j 项在第 c 帧完成；末帧的完成漏掉也不影响任何回传，不选
            done = [(t, j) for t, j in (n[1:] > n[:-1]).nonzero().tolist() if t + 2 < len(n)]
            if done and rng.random() < 0.5:
                t, j = rng.choice(done)
                fed[t + 2 :, j] -= 1                              # fed[c + 1] 起本应包含这次完成
            else:
                fed[rng.randrange(len(n)) :, rng.randrange(n.shape[1])] += 1
            fed.clamp_(0, MAX_COUNT)
        # 事件损失：完成帧前后 ±boundary 步（完成帧本身除外）不计，容忍标注时刻的误差。
        moved = torch.zeros(n.shape, dtype=torch.bool)
        moved[1:] = n[1:] != n[:-1]
        near = moved.clone()
        for shift in range(1, cfg.boundary + 1):
            near[shift:] |= moved[:-shift]
            near[:-shift] |= moved[shift:]
        masks["event_mask"] = (~near | moved)[:end]

        frames, stage, count = frames[:end], stage[:end], count[:end]

        # 标注关键帧 = 任一项计数变化的帧（子任务边界）；训练时随机偏移、丢弃（proposal 5.1）。
        # 训练时另外随机加入无关帧，模拟推理时的误触发；单独记录，关键帧总数超上限时先删它们。
        keyframes = torch.zeros(len(frames), dtype=torch.bool)
        jitter, drop, extra = (cfg.keyframe_jitter, cfg.keyframe_drop, cfg.keyframe_extra) if train else (0, 0, 0)
        for t in (t for t in changes if t < end):
            if rng.random() >= drop:
                keyframes[min(max(t + rng.randint(-jitter, jitter), 0), len(frames) - 1)] = True
        extra_keyframes = torch.tensor([rng.random() < extra for _ in frames], dtype=torch.bool) & ~keyframes

        # 示范：训练从示范池随机抽一条，关键帧时刻扰动、随机只留头部视角；验证用固定评估示范。
        demo_episode = rng.choice(list(self.splits[task]["pool"])) if train else self.splits[task]["eval_demo"]
        demo = json.loads((self.labels / task / f"episode{demo_episode}.json").read_text())
        jitter = cfg.demo_jitter if train else 0
        last_frame = demo.get("total_frames", demo["trace"][-1]["end"] + 1) - 1
        demo_frames = [min(max(item["frame"] + rng.randint(-jitter, jitter), 0), last_frame)
                       for item in demo["demo"]]
        cameras = demo.get("cameras", CAMERAS)
        cameras = cameras[:1] if train and rng.random() < cfg.drop_wrist else cameras
        demo_views = read_frames(self.source(demo), demo_frames, cameras)

        return {"task": task, "episode": episode, "instruction": record["instruction"], "steps": record["steps"],
                "demo": [{"step": item["step"], "views": views}
                         for item, views in zip(demo["demo"], demo_views, strict=True)],
                "frames": read_frames(self.source(record), [frame["index"] for frame in frames],
                                      record.get("cameras", CAMERAS)),
                "grad_from": grad_from, "stage": stage, "count": count, "fed": fed[:end],
                "progress": torch.tensor([frame["progress"] for frame in frames]), "total": len(record["trace"]),
                "keyframes": keyframes, "extra_keyframes": extra_keyframes, **masks}
