# ruff: noqa: RUF002, RUF003
"""按原视频划分训练/验证集，并创建使用现有 collate_fn 的 DataLoader。"""

import random

import torch
from torch.utils.data import DataLoader, Sampler, Subset

from qwen_training.dataset import VLMProgressDataset, collate_fn


# (输入 nowid, 输出 nowid)：每个 epoch 的目标百分比，合计 100%。
SAMPLING_PERCENTAGES = {
    (3, 1): 3.14,
    (3, 2): 7.08,
    (1, 2): 8.44,
    (2, 1): 9.45,
    (1, 3): 9.52,
    (2, 3): 12.37,
    (3, 4): 10.00,
    (1, 1): 10.00,
    (2, 2): 10.00,
    (3, 3): 10.00,
    (4, 4): 10.00,
}


class StateRatioSampler(Sampler):
    """按类别配额抽样；类内每轮不放回，抽完后重新打乱，总长度等于训练子集。"""

    def __init__(self, dataset, seed=0):
        self.num_samples = len(dataset)
        self.generator = torch.Generator().manual_seed(seed)
        self.groups = {pair: [] for pair in SAMPLING_PERCENTAGES}
        for index, source_index in enumerate(dataset.indices):
            sample = dataset.dataset.samples[source_index]
            pair = (sample["input_state"]["nowid"], sample["target"]["nowid"])
            self.groups[pair].append(index)  # 返回 Subset 内的位置，而非原始数据集下标。
        missing = [pair for pair, indices in self.groups.items() if not indices]
        if missing:
            raise ValueError(f"训练集缺少这些类别，无法按约定比例采样：{missing}")

        # 用 0.01% 为整数单位，先取整，再把余数分给小数部分最大的类别。
        scaled = {pair: self.num_samples * round(percent * 100)
                  for pair, percent in SAMPLING_PERCENTAGES.items()}
        self.quotas = {pair: value // 10000 for pair, value in scaled.items()}
        remaining = self.num_samples - sum(self.quotas.values())
        for pair in sorted(scaled, key=lambda p: scaled[p] % 10000, reverse=True)[:remaining]:
            self.quotas[pair] += 1

    def __iter__(self):
        sampled = []
        for pair, indices in self.groups.items():
            remaining = self.quotas[pair]
            while remaining:
                order = torch.randperm(len(indices), generator=self.generator).tolist()
                chosen = order[:remaining]
                sampled.extend(indices[i] for i in chosen)
                remaining -= len(chosen)
        order = torch.randperm(len(sampled), generator=self.generator).tolist()
        return iter([sampled[i] for i in order])

    def __len__(self):
        return self.num_samples


def build_dataloaders(index_files, batch_size=2, num_workers=2, val_fraction=0.1, seed=0):
    """返回 (train_loader, val_loader)，每批均为 (inputs, targets)。

    index_files：一个索引路径，或示范/恢复两组索引路径的列表。
    batch_size：每批样本数；最后不足一批的样本也保留。
    num_workers：每个 loader 的读取进程数；0 表示在主进程读取。
    val_fraction：每组数据中用于验证的视频比例，按视频数四舍五入。
    seed：控制视频划分和训练样本打乱；验证集顺序固定。

    同一个原视频的所有 episode 和采样窗口始终在同一侧。
    分组键包含 dataset 名称，两组数据里的 file-000 不会被混为同一视频。
    训练集按 SAMPLING_PERCENTAGES 分配每轮配额，类内随机抽样；验证集保持原分布。
    每轮抽样总数等于训练子集大小，整数配额与目标比例的误差小于一个样本。
    使用多进程时，在 if __name__ == "__main__" 内创建并迭代 loader。
    """
    dataset = VLMProgressDataset(index_files)
    videos_by_dataset = {}
    for sample in dataset.samples:
        videos_by_dataset.setdefault(sample["dataset"], set()).add(sample["video_id"])

    rng = random.Random(seed)
    val_videos = set()
    for name, video_ids in sorted(videos_by_dataset.items()):
        video_ids = sorted(video_ids)
        rng.shuffle(video_ids)
        count = int(len(video_ids) * val_fraction + 0.5)
        val_videos.update((name, video_id) for video_id in video_ids[:count])

    train_indices, val_indices = [], []
    for index, sample in enumerate(dataset.samples):
        key = (sample["dataset"], sample["video_id"])
        indices = val_indices if key in val_videos else train_indices
        indices.append(index)

    options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
        # spawn 避免子进程继承训练主进程的 CUDA 状态；每个 worker 预取一批。
        "multiprocessing_context": "spawn" if num_workers else None,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 1 if num_workers else None,
    }
    train_dataset = Subset(dataset, train_indices)
    # 样本打乱与 worker 启动各用一个随机数生成器，切换 worker 数不改变采样顺序。
    train_loader = DataLoader(
        train_dataset,
        sampler=StateRatioSampler(train_dataset, seed=seed),
        generator=torch.Generator().manual_seed(seed + 1), **options,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), shuffle=False,
        generator=torch.Generator().manual_seed(seed + 2), **options,
    )
    return train_loader, val_loader
