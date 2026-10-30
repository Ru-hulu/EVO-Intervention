"""读取 episode 窗口，训练时交换有效帧的两段，图像与标签按帧组同步排列。"""

from copy import copy
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from qwen_training.reference_images import load_window_images
from qwen_training.reference_io import FORMAT_VERSION, IMAGE_COUNT, format_frame_targets, make_reference_message
from qwen_training.reference_sampling import split_reference_indices, swap_valid_blocks


class ReferenceWindowDataset(Dataset):
    def __init__(self, index_files, shuffle_probability=0.0, seed=0):
        if isinstance(index_files, (str, Path)):
            index_files = [index_files]
        self.samples, self.base_dirs = [], []
        self.shuffle_probability, self.seed, self.epoch = shuffle_probability, seed, 0
        for name in index_files:
            path = Path(name).resolve()
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        sample = json.loads(line)
                        if sample.get("format_version") != FORMAT_VERSION:
                            raise ValueError(f"需要 {FORMAT_VERSION} 索引：{path}")
                        self.samples.append(sample)
                        self.base_dirs.append(path.parent)

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        source = self.samples[index]
        # 与 worker/rank 数量无关；同一轮同一样本可复现，下一轮重新决定是否交换。
        rng = random.Random(f"{self.seed}:{self.epoch}:{source['sample_id']}")
        sample = swap_valid_blocks(source, rng, self.shuffle_probability)
        return {
            "task": sample["task"], "frames": sample["frames"],
            "images": load_window_images(sample, self.base_dirs[index]),
            "targets": sample["targets"], "frame_order": sample["frame_order"],
        }


def collate_reference_windows(samples):
    """图片与消息占位符按同一帧组顺序展开，答案行与输入位置一一对应。"""
    if any(len(sample["images"]) != IMAGE_COUNT for sample in samples):
        raise ValueError("每个窗口必须有13张图像")
    return (
        {"messages": [make_reference_message(s["task"], s["frames"]) for s in samples],
         "images": [s["images"] for s in samples]},
        {"responses": [format_frame_targets(s["targets"]) for s in samples]},
    )


def build_reference_dataloaders(index_files, batch_size=2, num_workers=2, val_fraction=0.1,
                                seed=0, shuffle_probability=0.3):
    if not 0 <= shuffle_probability <= 1:
        raise ValueError("shuffle_probability 必须位于 [0,1]")
    dataset = ReferenceWindowDataset(index_files, seed=seed)
    train_indices, val_indices = split_reference_indices(dataset.samples, val_fraction, seed)
    # 共享只读索引，训练/验证使用不同的增强配置；不会把训练开关带入验证。
    train_dataset = copy(dataset)
    train_dataset.shuffle_probability = shuffle_probability
    train, val = Subset(train_dataset, train_indices), Subset(dataset, val_indices)
    options = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_reference_windows,
        multiprocessing_context="spawn" if num_workers else None,
        persistent_workers=False, prefetch_factor=1 if num_workers else None,
    )
    return (
        DataLoader(train, sampler=RandomSampler(train, generator=torch.Generator().manual_seed(seed)),
                   generator=torch.Generator().manual_seed(seed + 1), **options),
        DataLoader(val, shuffle=False, generator=torch.Generator().manual_seed(seed + 2), **options),
    )
