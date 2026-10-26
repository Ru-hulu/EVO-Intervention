"""加载逐帧三视角样本与冻结历史缓存；监督 PT 与模型输入分开返回。"""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

from mem_lite.datasets.m3_inputs import M3Collator

VIEW_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class M3Dataset(Dataset):
    def __init__(self, sample_files, history_dim=512):
        self.samples, self.episodes = [], []
        self.history_dim = history_dim
        self._history_cache = OrderedDict()
        signature = hashlib.sha256()
        self.history_source = None
        for source in sample_files:
            path = Path(source)
            if path.is_dir():
                path = path / "samples.jsonl"
            summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
            if summary["format_version"] != "mem-lite-m3-v1":
                raise ValueError("需要 M3 样本，不能把 M1/M2 样本作为进度标签")
            if summary["history_dim"] != history_dim:
                raise ValueError("历史缓存维度与 M3 history_dim 不一致")
            origin = summary["history_source"]
            if self.history_source is not None and origin != self.history_source:
                raise ValueError("同次训练不能混用不同第二阶段权重或构造规则的历史缓存")
            self.history_source = origin
            history_file = path.parent / "history.pt"
            if file_sha256(history_file) != summary["history_sha256"]:
                raise ValueError("history.pt 与构造记录不一致，请重新生成该视频样本")
            raw = path.read_bytes()
            # 路径不参与签名，允许同一份样本迁移到远程；内容和文件顺序必须相同。
            signature.update(bytes.fromhex(hashlib.sha256(raw).hexdigest()))
            signature.update(bytes.fromhex(summary["history_sha256"]))
            signature.update(json.dumps([summary["task_prompt"], origin], sort_keys=True).encode())
            episode = len(self.episodes)
            self.episodes.append((path.parent, summary))
            self.samples.extend((episode, json.loads(line)) for line in raw.splitlines() if line.strip())
        if not self.samples:
            raise ValueError("M3 样本集为空，请先准备带逐帧 PT 的三视角样本")
        self.signature = signature.hexdigest()

    def __len__(self):
        return len(self.samples)

    def _history(self, episode):
        if episode not in self._history_cache:
            directory, _ = self.episodes[episode]
            cache = torch.load(directory / "history.pt", map_location="cpu", weights_only=True)
            tokens, available = cache["history_tokens"], cache["available_from_frame"]
            if tokens.shape != (len(available), 128, self.history_dim) or not torch.isfinite(tokens).all():
                raise ValueError("历史缓存必须包含有限的 [事件数, 128, history_dim] tokens")
            if any(a >= b for a, b in zip(available, available[1:])):
                raise ValueError("历史可用帧号必须严格递增")
            self._history_cache[episode] = cache
            # 只缓存最近4条视频的小型历史张量，不让每个worker长期持有全数据集历史。
            if len(self._history_cache) > 4:
                self._history_cache.popitem(last=False)
        self._history_cache.move_to_end(episode)
        return self._history_cache[episode]

    def __getitem__(self, index):
        episode, sample = self.samples[index]
        directory, summary = self.episodes[episode]
        pt = float(sample["pt"])
        if not 0 <= pt <= 1:
            raise ValueError("PT 必须为 [0,1] 的有限数值，不自动补标签或截断数值")
        images = []
        for view in VIEW_ORDER:
            with Image.open(directory / sample["images"][view]) as image:
                images.append(image.convert("RGB"))
        cache = self._history(episode)
        history_index = sample["history_index"]
        available = cache["available_from_frame"]
        expected = sum(frame <= sample["frame_index"] for frame in available) - 1
        if history_index != (expected if expected >= 0 else None):
            raise ValueError("当前帧只能使用最新已可用的历史，不能读取未来事件或错用其他前缀")
        history = None if history_index is None else cache["history_tokens"][history_index].detach()
        return {"images": images, "task_prompt": summary["task_prompt"],
                "history_tokens": history, "target_pt": pt}


class Stage3Collator:
    def __init__(self, processor, history_dim=512, max_sequence_length=8192):
        self.inputs = M3Collator(processor, history_dim, max_sequence_length)

    def __call__(self, samples):
        return {"inputs": self.inputs(samples),
                "target_pt": torch.tensor([sample["target_pt"] for sample in samples], dtype=torch.float32)}
