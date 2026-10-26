# ruff: noqa: RUF002, RUF003
"""损失（proposal_progress 5.2），训练和验证共用。"""

import torch
import torch.nn.functional as F

LABELS = ("stage", "count", "progress", "stage_mask", "count_mask")


def autocast(cfg):
    """关掉类型转换缓存：头的权重最先在无梯度热身里被转成 bf16，缓存的副本没有梯度，之后的带梯度步会拿不到梯度。"""
    device = torch.device(cfg.device).type
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda", cache_enabled=False)


def sample_loss(model, outputs, sample, cfg):
    """L_STG + L_CNT + L_P + λ·L_interval，对样本中 grad_from 之后的各步取平均；边界附近不计 STG、CNT。"""
    logits = {name: torch.stack([out[name] for out in outputs]) for name in ("stage", "count", "progress")}
    labels = {name: sample[name][sample["grad_from"]:].to(model.special.device) for name in LABELS}
    stage_mask, count_mask = labels["stage_mask"], labels["count_mask"]
    stage = F.cross_entropy(logits["stage"][stage_mask], labels["stage"][stage_mask]) \
        if stage_mask.any() else logits["stage"].sum() * 0
    # 计数：每个 (帧, 项) 对 r 求和，再对 (帧, 项) 取平均。直接对 J×R 个输出取平均时，大量容易的负例
    # （如固定流程的"至少 2 次"）会把"至少 1 次"这类关键判断的梯度稀释约 R 倍。
    bce = F.binary_cross_entropy_with_logits(logits["count"], labels["count"], reduction="none")
    count = (bce * count_mask).sum() / count_mask.any(-1).sum().clamp(min=1)

    # p 的标签摊到相邻两个分箱上（线性插值），对分箱分布做交叉熵。
    position = labels["progress"].clamp(0, 1) * (cfg.progress_bins - 1)
    low = position.floor().long().clamp(max=cfg.progress_bins - 2)
    high = (position - low)[:, None]
    target = torch.zeros(len(low), cfg.progress_bins, device=low.device)
    target.scatter_(1, low[:, None], 1 - high).scatter_add_(1, low[:, None] + 1, high)
    progress = -(target * F.log_softmax(logits["progress"], -1)).sum(-1).mean()

    # 只在训练时用标注的 K 约束 p 落在 [k/K, (k+1)/K]；k = 各项期望次数之和 = Σ_j Σ_r P(n_j ≥ r)，不反传到 [CNT]。
    p = (F.softmax(logits["progress"], -1) * model.bins).sum(-1)
    k, total = logits["count"].detach().sigmoid().sum((-2, -1)), sample["total"]
    interval = (F.relu(k / total - p) + F.relu(p - (k + 1) / total)).mean()
    loss = stage + count + progress + cfg.interval_weight * interval
    return loss, {"stage": stage.item(), "count": count.item(), "progress": progress.item(),
                  "interval": interval.item()}
