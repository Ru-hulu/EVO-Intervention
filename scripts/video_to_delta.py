#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""逐帧计算 Qwen 视觉表征的 L2 差值，并附上 JSON 中的阶段描述。

使用远程已有的 Qwen 环境（Transformers 4.57.6、PyTorch、Accelerate、PyAV）：
    python video_to_delta.py

默认输入 /root/video/file-xxx.mp4（30fps 原视频）、/root/lable/file-xxx.JSON，
输出 /root/pt/file-xxx-delta.txt。第一行是 0，其余行是「delta 当前帧描述」。
解码帧序号从 0 开始，直接匹配 JSON 的 30FPSstart_frame / 30FPSend_frame。
只计算原始 delta，不计算归一化进度，也不在 delta 上添加 epsilon。
"""

import argparse
import json
from pathlib import Path

from accelerate import init_empty_weights
import av
from safetensors import safe_open
import torch
from transformers import AutoConfig
from transformers import AutoImageProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


class QwenDelta:
    """持有视觉编码器，并缓存上一张图像的表征。"""

    def __init__(self, model_dir: Path, device: str = "cuda", max_pixels: int = 262144):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(
            model_dir, local_files_only=True, use_fast=True, min_pixels=4096, max_pixels=max_pixels
        )
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True).vision_config
        with init_empty_weights():
            self.model = Qwen3VLVisionModel._from_config(config, attn_implementation="sdpa")  # noqa: SLF001

        # 只从分片中读取 model.visual.*，不加载语言模型的权重。
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        prefix = "model.visual."
        vision_weights = {key: shard for key, shard in index["weight_map"].items() if key.startswith(prefix)}
        state_dict = {}
        for shard in sorted(set(vision_weights.values())):
            with safe_open(model_dir / shard, framework="pt", device="cpu") as weights:
                for key, filename in vision_weights.items():
                    if filename == shard:
                        state_dict[key.removeprefix(prefix)] = weights.get_tensor(key)
        self.model.load_state_dict(state_dict, assign=True)
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.last_image = None
        self.last_phi = None

    @torch.inference_mode()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """输入 RGB、CHW、uint8（0–255）；输出一个 FP32 视觉向量。"""
        inputs = self.processor(images=image, return_tensors="pt")
        tokens, _ = self.model(
            inputs["pixel_values"].to(device=self.device, dtype=self.dtype),
            grid_thw=inputs["image_grid_thw"].to(self.device),
        )
        # 使用视觉模块经 Merger 输出的 token；4B 模型池化后是 2560 维。
        return tokens.float().mean(dim=0)

    def image2delta(self, image1: torch.Tensor, image2: torch.Tensor) -> float:
        """两张 RGB tensor → ||phi(image2) - phi(image1)||₂；输入不能原地修改。"""
        phi1 = self.last_phi if image1 is self.last_image else self.encode(image1)
        phi2 = self.encode(image2)
        self.last_image, self.last_phi = image2, phi2
        return torch.linalg.vector_norm(phi2 - phi1).item()


def load_labels(json_path: Path) -> dict[int, str]:
    """建立 30fps 帧编号到阶段描述的映射，区间包含首尾帧。"""
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    labels = {}
    for task in data["tasks"]:
        for episode in task["episodes"]:
            for subtask in episode["subtasks"]:
                start = subtask["30FPSstart_frame"]
                end = subtask["30FPSend_frame"]
                for frame_index in range(start, end + 1):
                    labels[frame_index] = subtask["name"]
    return labels


def process_video(video_path: Path, json_path: Path, output_path: Path, encoder: QwenDelta) -> None:
    # 处理完成后再替换目标文件，避免中断时留下看似完整的 txt。
    temporary_path = output_path.with_suffix(".txt.tmp")
    with av.open(str(video_path)) as video:
        labels = load_labels(json_path)
        previous = None
        frame_count = 0
        with temporary_path.open("w", encoding="utf-8") as output:
            for frame_index, frame in enumerate(video.decode(video=0)):
                current = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                if previous is None:
                    output.write("0\n")
                else:
                    delta = encoder.image2delta(previous, current)
                    description = labels.get(frame_index, "未标注")
                    output.write(f"{delta:.6f} {description}\n")
                previous = current
                frame_count += 1
    temporary_path.replace(output_path)
    print(f"完成 {output_path.name}：{frame_count} 帧 / {frame_count} 行", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("/root/video"))
    parser.add_argument("--label-dir", type=Path, default=Path("/root/lable"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/pt"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/root/autodl-tmp/qwen3-vl-ft/models/Qwen3-VL-4B-Instruct")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pixels", type=int, default=262144)
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("file-*.mp4"))
    if not videos:
        raise FileNotFoundError(f"{args.video_dir} 下没有 file-*.mp4")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = QwenDelta(args.model_dir, args.device, args.max_pixels)
    for video_path in videos:
        process_video(
            video_path,
            args.label_dir / f"{video_path.stem}.JSON",
            args.output_dir / f"{video_path.stem}-delta.txt",
            encoder,
        )


if __name__ == "__main__":
    main()
