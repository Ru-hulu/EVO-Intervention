"""M2 累计队列加载、图像去重及第二阶段两条监督分支的批次组织。"""

import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from mem_lite.datasets.m1_dataset import M1Collator, M1Dataset


class M2Dataset(Dataset):
    def __init__(self, sample_files):
        self.samples = []
        for source in sample_files:
            path = Path(source)
            if path.is_dir():
                path = path / "samples.jsonl"
            with path.open(encoding="utf-8") as stream:
                self.samples.extend((path.parent, json.loads(line)) for line in stream if line.strip())
        if not self.samples:
            raise ValueError("M2 样本集为空，请先构造事件队列 samples.jsonl")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        directory, sample = self.samples[index]
        events = sample["inputs"]["events"]
        if [target["prefix_length"] for target in sample["targets"]] != list(range(1, len(events) + 1)):
            raise ValueError("M2 样本必须包含每个非空累计前缀的一条状态标签")
        images, image_indices, event_indices = [], {}, []
        for event in events:
            indices = []
            for frame in event["frames"]:
                path = frame["image"]
                if path not in image_indices:
                    image_indices[path] = len(images)
                    with Image.open(directory / path) as image:
                        images.append(image.convert("RGB"))
                indices.append(image_indices[path])
            event_indices.append(indices)
        return {
            "images": images, "event_frame_indices": event_indices,
            "state_descriptions": [target["state_description"] for target in sample["targets"]],
        }


class M2Collator:
    def __init__(self, processor):
        self.image_processor = processor.image_processor

    def __call__(self, samples):
        images, event_indices, counts, descriptions = [], [], [], []
        for sample in samples:
            offset = len(images)
            event_indices.extend([[offset + index for index in event] for event in sample["event_frame_indices"]])
            images.extend(sample["images"])
            counts.append(len(sample["event_frame_indices"]))
            descriptions.extend(sample["state_descriptions"])
        vision = self.image_processor(images=images, return_tensors="pt")
        return {
            "pixel_values": vision["pixel_values"], "image_grid_thw": vision["image_grid_thw"],
            "event_frame_indices": event_indices, "queue_event_counts": counts,
            "state_descriptions": descriptions,
        }


class Stage2Dataset(Dataset):
    """以 M2 队列数定义 epoch；每队列配一条可复现随机抽取的 M1 样本。"""

    def __init__(self, m2_sample_files, m1_sample_files, seed=42):
        self.memory = M2Dataset(m2_sample_files)
        self.segmentation = M1Dataset(m1_sample_files)
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        return {"memory": self.memory[index], "m1": self.segmentation[rng.randrange(len(self.segmentation))]}


class Stage2Collator:
    def __init__(self, processor, max_sequence_length=8192):
        self.m1 = M1Collator(processor, max_sequence_length)
        self.memory = M2Collator(processor)

    def __call__(self, samples):
        return {"m1": self.m1([sample["m1"] for sample in samples]),
                "memory": self.memory([sample["memory"] for sample in samples])}
