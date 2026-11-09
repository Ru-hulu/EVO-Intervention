# ruff: noqa: RUF002, RUF003
"""样本只有任务指令、帧和 p：不读示范，不用子任务边界。

p 是帧在这条轨迹里的位置比例，随帧走：停顿重复某一帧时，p 也跟着重复。
带 progress_source=dense_annotation 的轨迹按源帧读取完整标注，保留冻结和突变。
旧脚本失败/重试轨迹（progress_label 为 retained_progress_v4）沿用稀疏标签插值。
"""

import json
import random
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FIXED = "fixed"     # 固定流程任务的 kind，repeat_boost 不加倍
AGIBOT = {"C13", "C14", "O05", "O06", "O07"}    # AgiBot 采集的任务，画面上限是 agibot_max_*


def read_video(path, indices, box=None):
    """按原视频帧号读 RGB；保留请求顺序和重复帧。box=(宽, 高) 时解码时就按面积平均缩到这个范围内，不放大。"""
    import av

    wanted, decoded = set(indices), {}
    last = max(wanted)
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                if box is not None:
                    scale = min(1, box[0] / frame.width, box[1] / frame.height)
                    frame = frame.reformat(width=round(frame.width * scale), height=round(frame.height * scale),
                                           format="rgb24", interpolation="AREA")
                decoded[index] = frame.to_ndarray(format="rgb24")
            if index >= last:
                break
    if wanted - decoded.keys():
        raise ValueError(f"{path}: 无法读取帧 {sorted(wanted - decoded.keys())}")
    return [decoded[index] for index in indices]


def read_frames(paths, indices, cameras, box=None):
    """[[每个视角的 RGB 数组] for 每个帧号]，paths: {视角: mp4 路径}。"""
    views = [read_video(paths[camera], indices, box) for camera in cameras]
    return [[view[t] for view in views] for t in range(len(indices))]


def add_pauses(n, cfg, rng):
    """时间增强：以 pause_prob 在 1–3 处把某一帧重复 5–20 步，返回帧下标序列。"""
    order = list(range(n))
    if rng.random() < cfg.pause_prob:
        for _ in range(rng.randint(1, 3)):
            t = rng.randrange(len(order))
            order[t:t] = [order[t]] * rng.randint(5, 20)
    return order


def speed_positions(n, speed):
    """n 帧的轨迹按 speed 倍速重新取帧时，每一步落在原帧序的位置（可为小数），末帧总在最后。只在验证里用。"""
    return [*np.arange(0, n - 1, speed).tolist(), n - 1]


def full_ranges(length, segment, rng):
    """把整条轨迹切成首尾相接的段，全部带梯度；首段长度随机，段边界不会总落在同样的帧上。"""
    return list(pairwise([0, *range(rng.randint(1, segment), length, segment), length]))


def reorder(sample, order):
    """按帧下标序列重排一条样本。"""
    return {**sample, "frames": [sample["frames"][i] for i in order], "progress": sample["progress"][order],
            "positions": sample["positions"][order]}


def keep_sample(sample):
    return sample


class Trajectories:
    def __init__(self, labels, data, split, suites=None):
        self.labels, self.data = Path(labels), Path(data)
        splits = json.loads((self.labels / "_splits.json").read_text())
        self.tasks = json.loads((self.labels / "_tasks.json").read_text())
        self.episodes = [(task, episode) for task in sorted(splits) for episode in splits[task][split]
                         if (not suites or self.tasks[task]["suite"] in suites)
                         and (self.labels / task / f"episode{episode}.json").exists()]
        if not self.episodes:
            raise ValueError(f"{self.labels} 下没有 {split} 划分的标签")

    def __len__(self):
        return len(self.episodes)

    def sample(self, index, cfg, rng=None, speed=None):
        """rng 为 None 时是验证样本：整条轨迹、不做增强，speed 固定倍速。训练时只有随机停顿。

        positions 是每一步在原帧序的位置，比较内容和步数时用（clock_test）。
        """
        task, episode = self.episodes[index]
        record = json.loads((self.labels / task / f"episode{episode}.json").read_text())
        indices = [frame["index"] for frame in record["frames"]]
        positions = np.array(range(len(indices)) if speed is None else speed_positions(len(indices), speed), float)
        if rng is not None:
            positions = positions[add_pauses(len(positions), cfg, rng)]
        source = np.interp(positions, np.arange(len(indices)), indices).round().astype(int)
        if record.get("progress_source") == "dense_annotation":
            dense = np.asarray(record["dense_progress"], dtype=float)
            if dense.shape != (record["total_frames"],) or not np.isfinite(dense).all() \
                    or np.any((dense < 0) | (dense > 1)):
                raise ValueError("dense_annotation 必须包含每个源视频帧的有效 [0,1] progress")
            progress = dense[source]
        elif record.get("progress_label") == "retained_progress_v4":
            # 脚本失败/重试轨迹：p 是标注的逐帧 v4 真值，失败时会回落，不按位置比例算。
            progress = np.interp(source, indices, [frame["progress"] for frame in record["frames"]])
        else:
            progress = (source - indices[0]) / (indices[-1] - indices[0])
        paths = {camera: self.data / path for camera, path in record["source"].items()}
        box = (cfg.agibot_max_width, cfg.agibot_max_height) if self.tasks[task]["task_id"] in AGIBOT else None
        return {"task": task, "episode": episode, "instruction": record["instruction"],
                "frames": read_frames(paths, source.tolist(), record["cameras"], box),
                "progress": torch.tensor(progress, dtype=torch.float32), "positions": positions,
                "segment_ranges": None if rng is None else full_ranges(len(positions), cfg.segment, rng)}


class EpochSamples(Dataset):
    """打乱的一个 epoch，每条轨迹至少出现一次。

    每个任务的样本数是最大任务轨迹数的 repeat_boost 倍，固定流程任务是 1 倍；多出来的次数均摊到任务的各条轨迹上。
    每个位置有自己的随机种子，增强结果和 worker 的预取顺序无关。
    """

    def __init__(self, dataset, cfg, epoch, offset=0):
        self.dataset, self.cfg, self.epoch, self.offset = dataset, cfg, epoch, offset
        groups = {}
        for index, (task, _) in enumerate(dataset.episodes):
            groups.setdefault(task, []).append(index)
        rng = random.Random(f"{cfg.seed}:epoch:{epoch}")
        largest = max(map(len, groups.values()))
        order = []
        for task, indices in groups.items():
            boost = 1 if dataset.tasks[task]["kind"] == FIXED else cfg.repeat_boost
            whole, rest = divmod(max(len(indices), round(largest * boost)), len(indices))
            order.extend(indices * whole + rng.sample(indices, rest))
        rng.shuffle(order)
        self.order = order

    def __len__(self):
        return len(self.order) - self.offset

    def __getitem__(self, position):
        position += self.offset
        rng = random.Random(f"{self.cfg.seed}:epoch:{self.epoch}:sample:{position}")
        return self.dataset.sample(self.order[position], self.cfg, rng)
