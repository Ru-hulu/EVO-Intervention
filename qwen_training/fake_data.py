"""固定的三视角历史假图像与意图、阶段文本，供完整视觉训练链路使用。"""
import torch

from qwen_training.prompts import TASK_INTENT, build_training_examples


def fakefunc(batch_size=3, num_frames=6, image_size=128, seed=0):
    """返回 (inputs, targets)，按顺序循环取保持、切换、回滚三种样本。

    inputs["messages"]：按正式训练的六种 prompt/记忆组合构造。
    inputs["images"]：CPU uint8 [B,3,L,3,H,W]，三视角依次为顶部、左侧、右侧，时间从旧到新。
        L=num_frames，包含当前帧；H=W=image_size，所有图片尺寸相同。
    inputs["frame_mask"]：bool [B,3,L]，此处生成的帧全部有效。
    targets["responses"]：每个样本的任务意图和当前阶段状态答案。
    targets["progress"]：float32 [B]，单独监督进度回归头。
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
    states, state_pairs, target_states = [], [], []
    for i in range(batch_size):
        previous, current = samples[i % len(samples)]
        states.append(previous)
        state_pairs.append((previous["nowid"], current["nowid"]))
        target_states.append({**current, "intent": TASK_INTENT})

    examples = build_training_examples(states, target_states, seed=seed)
    cases = torch.arange(batch_size) % 3
    generator = torch.Generator().manual_seed(seed)
    inputs = {
        "messages": examples["messages"],
        "input_states": states,
        "target_states": target_states,
        "state_pairs": state_pairs,
        "combination_ids": examples["combination_ids"],
        # 固定种子生成三组历史图像；重复调用使用相同样本，便于检查训练链路。
        "images": torch.randint(
            0, 256, (3, 3, num_frames, 3, image_size, image_size),
            generator=generator, dtype=torch.uint8,
        )[cases],  # [B,3,L,3,H,W]，由图像预处理器负责转换。
        "frame_mask": torch.ones(batch_size, 3, num_frames, dtype=torch.bool),  # [B,3,L]
    }
    targets = {
        "responses": examples["responses"],
        "progress": torch.tensor(examples["progress"], dtype=torch.float32),
        "progress_mask": torch.tensor(examples["progress_mask"], dtype=torch.bool),
    }
    return inputs, targets
