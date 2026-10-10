# ruff: noqa: RUF002, RUF003
"""读取已有六帧三视角索引，并整理成现有模型所需的批量输入。"""

import json
from pathlib import Path

import av
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from qwen_training.prompts import SYSTEM_PROMPT


AUGMENTED_PAIRS = {(3, 1), (3, 2), (1, 2), (2, 1)}
COLOR_FACTOR_RANGE = (0.95, 1.05)  # 亮度、对比度、饱和度均仅扰动 ±5%。


class VLMProgressDataset(Dataset):
    """index_files 为一个 samples.jsonl 路径，或两组数据的索引路径列表。

    直接使用索引里的视频路径、帧号和标签；视频应为已对齐的固定帧率视频，
    所有图像尺寸相同。帧号从 0 开始，null 表示左侧空白历史帧。
    初始化只读取索引，取样时才解码视频；每次读取后关闭视频文件。
    """

    def __init__(self, index_files):
        if isinstance(index_files, (str, Path)):
            index_files = [index_files]
        self.samples = []
        for path in index_files:
            with Path(path).open(encoding="utf-8") as handle:
                self.samples.extend(json.loads(line) for line in handle if line.strip())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        indices = sample["frame_indices"]  # 长度 6，从旧到新，末帧是真实帧。
        wanted = {frame for frame in indices if frame is not None}
        images = []
        for view in ("top", "left", "right"):
            decoded = {}
            with av.open(sample["video_paths"][view]) as video:
                stream = video.streams.video[0]
                fps = stream.average_rate
                start_time = stream.start_time or 0
                # 跳到首个目标帧之前的关键帧，再按时间戳定位原视频帧号。
                timestamp = start_time + int(min(wanted) / fps / stream.time_base)
                video.seek(timestamp, stream=stream, backward=True)
                for frame in video.decode(stream):
                    frame_index = round((frame.pts - start_time) * stream.time_base * fps)
                    if frame_index in wanted:
                        rgb = frame.to_ndarray(format="rgb24")  # [H,W,3]，uint8。
                        decoded[frame_index] = torch.from_numpy(rgb).permute(2, 0, 1)  # [3,H,W]。
                    if frame_index >= indices[-1]:
                        break
            blank = torch.zeros_like(decoded[indices[-1]])
            images.append(torch.stack([blank if i is None else decoded[i] for i in indices]))

        return {
            "sample_id": sample["sample_id"],
            "images": torch.stack(images),  # CPU uint8 [3,6,3,H,W]：视角、时间、RGB、高、宽。
            "frame_mask": torch.tensor(sample["frame_mask"], dtype=torch.bool),  # [3,6]。
            "input_state": sample["input_state"],  # 倒数第二个采样时刻的状态。
            "target": sample["target"],  # 最后一个采样时刻的状态及已计算的 progress。
        }


def augment_training_images(images, messages, responses, seed):
    """训练专用：按状态对原地更新 CPU uint8 [B,3,6,3,H,W] 图像，返回 images。

    每个目标样本的 18 张图共享三个增强系数，不同样本各自采样。
    seed 使用训练种子 + 当前 step；独立生成器不影响模型随机状态。
    图像尺寸、相机与时间顺序不变，messages / responses 仅用于读取类别。
    """
    generator = torch.Generator().manual_seed(seed)
    low, high = COLOR_FACTOR_RANGE
    for i, (chat, response) in enumerate(zip(messages, responses)):
        pair = (json.loads(chat[-1]["content"])["nowid"], json.loads(response)["nowid"])
        if pair not in AUGMENTED_PAIRS:
            continue
        brightness, contrast, saturation = (
            low + (high - low) * torch.rand(3, generator=generator)
        ).tolist()
        frames = images[i]  # [3,6,3,H,W]，三个视角、六个时刻使用相同参数。
        frames = TF.adjust_brightness(frames, brightness)
        frames = TF.adjust_contrast(frames, contrast)
        images[i] = TF.adjust_saturation(frames, saturation)
    return images


def collate_fn(samples):
    """单样本列表 → (inputs, targets)，与 fakefunc 的批量接口一致。"""
    inputs = {
        "images": torch.stack([sample["images"] for sample in samples]),  # [B,3,6,3,H,W]。
        "frame_mask": torch.stack([sample["frame_mask"] for sample in samples]),  # [B,3,6]。
        "messages": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(sample["input_state"], ensure_ascii=False)},
            ]
            for sample in samples
        ],
    }
    targets = {
        "responses": [json.dumps(sample["target"], ensure_ascii=False) for sample in samples],
    }
    return inputs, targets
