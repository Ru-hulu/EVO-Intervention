"""按视频分组、组内按时间推进；复用现有图像、PT 和 Forest 历史快照。"""

import math
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler

from .m3_inputs import M3Collator


class EpisodeWindowSampler(Sampler):
    """batch_size 是每卡并行视频数；仅打乱视频，不打乱视频内部的帧。

    所有 rank 使用相同组边界和窗口数。尾部空槽只参与同步，不重复监督样本。
    每组结束才允许 optimizer.step/checkpoint，恢复时无需保存过期的递归状态。
    """

    def __init__(self, dataset, batch_size, window_size=8, rank=0, world_size=1, seed=42):
        if min(batch_size, window_size, world_size) < 1 or not 0 <= rank < world_size:
            raise ValueError("视频 batch、窗口和 world_size 必须为正，rank 必须有效")
        episodes = defaultdict(list)
        for index, (episode, _, _) in enumerate(dataset.samples):
            episodes[episode].append(index)
        self.episodes = list(episodes.values())
        self.batch_size, self.window_size = batch_size, window_size
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def groups(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.episodes), generator=generator).tolist()
        width = self.batch_size * self.world_size
        for start in range(0, len(order), width):
            group = [self.episodes[i] for i in order[start:start + width]]
            yield group + [[] for _ in range(width - len(group))]

    def __len__(self):
        return sum(math.ceil(max(map(len, group)) / self.window_size) for group in self.groups())

    def __iter__(self):
        for group in self.groups():
            length = max(map(len, group))
            local = group[self.rank * self.batch_size:(self.rank + 1) * self.batch_size]
            for start in range(0, length, self.window_size):
                stop = min(start + self.window_size, length)
                yield {"indices": [[episode[t] if t < len(episode) else None for episode in local]
                                   for t in range(start, stop)],
                       "reset": start == 0, "group_end": stop == length}


class EpisodeWindows(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, window):
        # 空槽仍提供合法视觉输入，sample_valid 会阻止其状态更新与 loss。
        reference = next((i for row in window["indices"] for i in row if i is not None), 0)
        dummy = None
        frames = []
        for row in window["indices"]:
            samples = []
            for index in row:
                if index is not None:
                    samples.append(self.dataset[index])
                else:
                    if dummy is None:
                        dummy = {**self.dataset[reference], "history_tokens": None, "target_pt": 0.0}
                    samples.append(dummy)
            frames.append(samples)
        return {**window, "frames": frames}


class SequenceCollator:
    def __init__(self, processor, history_dim, max_sequence_length, history_capacity):
        self.inputs = M3Collator(processor, history_dim, max_sequence_length, history_capacity)

    def __call__(self, window):
        inputs, targets = [], []
        for row, samples in zip(window["indices"], window["frames"], strict=True):
            batch = self.inputs(samples)
            batch["sample_valid"] = torch.tensor([i is not None for i in row])
            inputs.append(batch)
            targets.extend(sample["target_pt"] for sample in samples)
        return {"inputs": inputs, "target_pt": torch.tensor(targets, dtype=torch.float32),
                "reset": window["reset"], "group_end": window["group_end"]}
