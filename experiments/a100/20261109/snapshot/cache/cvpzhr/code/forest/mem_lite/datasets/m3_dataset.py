"""按需读取三视角观测及 StreamForest 历史快照，监督与模型输入分开。"""

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from mem_lite.datasets.m3_inputs import M3Collator

VIEW_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DATA_FORMAT = "streamforest-m3-data-v1"


class M3Dataset(Dataset):
    def __init__(self, sample_files, history_dim=None, load_history=True):
        self.samples, self.episodes = [], []
        self.history_dim, self.history_capacity, self.history_source = history_dim, None, None
        self.load_history = load_history
        signature = hashlib.sha256()
        for source in sample_files:
            path = Path(source)
            if path.is_dir():
                path /= "samples.jsonl"
            summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
            if summary["format_version"] != DATA_FORMAT:
                raise ValueError("需要 StreamForest 样本，请重新构造旧历史缓存")
            for attr in ("history_dim", "history_capacity", "history_source"):
                expected = getattr(self, attr)
                if expected is not None and expected != summary[attr]:
                    raise ValueError(f"样本的 {attr} 不一致，不能混用不同记忆配置或视觉权重")
                setattr(self, attr, summary[attr])
            raw = path.read_bytes()
            signature.update(raw)
            signature.update(json.dumps(summary, sort_keys=True).encode())
            episode = len(self.episodes)
            self.episodes.append((path.parent, summary))
            previous = None
            for line in raw.splitlines():
                if not line.strip():
                    continue
                sample = json.loads(line)
                if previous is not None and sample["frame_index"] <= previous:
                    raise ValueError("同一视频的样本帧号必须递增")
                self.samples.append((episode, sample, previous))
                previous = sample["frame_index"]
        if not self.samples:
            raise ValueError("M3 样本集为空")
        self.signature = signature.hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode, sample, previous = self.samples[index]
        directory, summary = self.episodes[episode]
        pt = float(sample["pt"])
        if not 0 <= pt <= 1:
            raise ValueError("PT 必须为 [0,1] 的有限数值")
        images = []
        for view in VIEW_ORDER:
            with Image.open(directory / sample["images"][view]) as image:
                images.append(image.convert("RGB"))
        history = None
        if self.load_history:
            if (sample["history"] is None) != (previous is None):
                raise ValueError("只能使用上一观测帧形成的最新历史，首帧必须为空")
            if sample["history"] is not None:
                cache = torch.load(directory / sample["history"], map_location="cpu", weights_only=True)
                if cache["through_frame"] != previous:
                    raise ValueError("历史必须来自上一观测帧，不能读取未来或过期快照")
                history = cache["tokens"].detach()
        return {"images": images, "task_prompt": summary["task_prompt"], "history_tokens": history,
                "target_pt": pt, "frame_index": sample["frame_index"], "episode_id": str(directory.resolve())}


class Stage3Collator:
    def __init__(self, processor, history_dim, max_sequence_length=8192, history_capacity=1008):
        self.inputs = M3Collator(processor, history_dim, max_sequence_length, history_capacity)

    def __call__(self, samples):
        return {"inputs": self.inputs(samples),
                "target_pt": torch.tensor([sample["target_pt"] for sample in samples], dtype=torch.float32)}
