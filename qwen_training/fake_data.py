"""固定的 JSON 对话与视觉占位表征，供 Qwen 文本生成训练使用。"""
import json
import torch

SYSTEM_PROMPT = (
    "输入 JSON 表示上一次阶段更新时的状态。结合当前融合三视角与历史信息的视觉表征，"
    "判断当前任务状态与进度。"
    "任务 ID：1=拿起套筒，2=拿起螺钉，3=将螺钉和套筒对齐并插入，4=将装配好的组件放入盒子。"
    "nowid 是当前阶段；finishid 是当前仍有效的已完成任务；restid 是除当前阶段外仍需执行的任务。"
    "异常恢复时允许已完成任务失效、进度下降。"
    "仅输出合法 JSON：nowid 为整数，finishid 和 restid 为整数数组，progress 为 0 到 1 的数值。"
)

def fakefunc(batch_size=3, num_tokens=16, feature_dim=128, seed=0, device="cpu"):
    """返回 (inputs, targets)，按顺序循环取保持、切换、回滚三种样本。

    inputs["messages"]：每个样本的 system 指令和 user JSON（上次更新状态）。
    targets["responses"]：每个样本的 JSON 答案（当前状态与进度）。
    训练时将答案追加为 assistant 消息；推理时只使用 inputs。
    visual_features [B,N,D] 是已融合三视角与历史的占位表征，不写入 JSON。
    """
    # 每对分别是上次更新状态、当前答案；ID 沿用标注的 1..4。
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
        # 固定种子生成三个特征模板，使重复调用可以用于小样本拟合。
        "visual_features": torch.randn(
            3, num_tokens, feature_dim, generator=generator
        )[cases].to(device),  # [B,N,D]
        "visual_mask": torch.ones(batch_size, num_tokens, dtype=torch.bool, device=device),  # [B,N]
    }
    targets = {"responses": responses}
    return inputs, targets
