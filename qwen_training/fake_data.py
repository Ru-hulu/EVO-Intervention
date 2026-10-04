"""固定的三视角历史假图像与 JSON 对话，供完整视觉训练链路使用。"""
import json
import torch

from qwen_training.prompts import SYSTEM_PROMPT


def fakefunc(batch_size=3, num_frames=3, image_size=128, seed=0):
    """返回 (inputs, targets)，按顺序循环取保持、切换、回滚三种样本。

    inputs["messages"]：每个样本的 system 指令和 user JSON（上一观测时刻的状态）。
    inputs["images"]：CPU uint8 [B,3,L,3,H,W]，三视角依次为顶部、左侧、右侧，时间从旧到新。
        L=num_frames，包含当前帧；H=W=image_size，所有图片尺寸相同。
    inputs["frame_mask"]：bool [B,3,L]，此处生成的帧全部有效。
    targets["responses"]：每个样本的 JSON 答案（当前状态与进度）。
    图像先交给 model.encode_history()，得到视觉特征后再和 messages 一起送入 VLM。
    """
    # 每对分别是上一观测时刻的状态、当前答案；ID 沿用标注的 1..4。
    samples = [
        (
            {"nowid": 2, "finishid": [1], "restid": [3, 4]},
            {"nowid": 2, "finishid": [1], "restid": [3, 4], "progress": 0.4},
        ),
        (
            {"nowid": 2, "finishid": [1], "restid": [3, 4]},
            {"nowid": 3, "finishid": [1, 2], "restid": [4], "progress": 0.6},
        ),
        (
            {"nowid": 3, "finishid": [1, 2], "restid": [4]},
            {"nowid": 2, "finishid": [1], "restid": [3, 4], "progress": 0.3},
        ),
    ]
    messages, responses = [], []
    for i in range(batch_size):
        previous, current = samples[i % len(samples)]
        messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(previous, ensure_ascii=False)},
        ])
        responses.append(json.dumps(current, ensure_ascii=False))

    cases = torch.arange(batch_size) % 3
    generator = torch.Generator().manual_seed(seed)
    inputs = {
        "messages": messages,
        # 固定种子生成三组历史图像；重复调用使用相同样本，便于检查训练链路。
        "images": torch.randint(
            0, 256, (3, 3, num_frames, 3, image_size, image_size),
            generator=generator, dtype=torch.uint8,
        )[cases],  # [B,3,L,3,H,W]，由图像预处理器负责转换。
        "frame_mask": torch.ones(batch_size, 3, num_frames, dtype=torch.bool),  # [B,3,L]
    }
    targets = {"responses": responses}
    return inputs, targets
