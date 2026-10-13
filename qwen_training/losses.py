"""进度监督只覆盖有任务 prompt 的样本。"""

import torch


def masked_progress_mse(prediction, target, mask=None):
    valid = (torch.ones_like(prediction, dtype=torch.bool) if mask is None
             else mask.to(device=prediction.device, dtype=torch.bool))
    count = valid.sum()
    # 空选择的 sum 仍连接计算图；整批无 prompt 时回归头得到零梯度。
    loss = (prediction[valid] - target.to(prediction)[valid]).square().sum() / count.clamp_min(1)
    return loss, count
