# ruff: noqa: RUF002, RUF003
"""损失（proposal_progress 5.2），训练和验证共用。"""

import torch
import torch.nn.functional as F

LABELS = ("stage", "count", "progress", "stage_mask", "count_mask", "event_mask")


def autocast(cfg):
    """关掉类型转换缓存：头的权重最先在无梯度热身里被转成 bf16，缓存的副本没有梯度，之后的带梯度步会拿不到梯度。"""
    device = torch.device(cfg.device).type
    return torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda", cache_enabled=False)


def event_loss(count_logits, count_labels):
    """计数分数 g_j 两步内的变化 Δ = g_j(t) − g_j(t−2)（与阈值无关，用 r = 1 的 logit 算）：

    - 这两步内第 j 项次数变了：Δ 至少要朝变化的方向跳 1（约一个阈值间隔），损失 relu(1 − Δ·sign)；
    - 没变：Δ 应为 0，损失 |Δ|。L1 让分数分段恒定，按时间缓慢上升的斜坡在每一步都要付代价。
    两步的窗口容忍完成时刻 ±1 步的歧义。变了的 (t, j) 远少于没变的，两类各自取平均再相加，免得跳变的
    要求被稀释；某一类不存在时那一项为 0。count_logits [T, J, R]，count_labels [T, J, R]；不足 3 步时为 0。
    """
    if len(count_logits) < 3:
        return count_logits.sum() * 0
    score, count = count_logits[..., 0], count_labels.sum(-1)
    jump, change = score[2:] - score[:-2], (count[2:] - count[:-2]).sign()
    still, moved = change == 0, change != 0
    loss = jump.abs()[still].mean() if still.any() else jump.sum() * 0
    if moved.any():
        loss = loss + F.relu(1 - jump * change)[moved].mean()
    return loss


def change_loss(logits, target, mask):
    """count_mode=event 的 CNT 损失：每帧每项次数变化 {−1, 0, +1} 的交叉熵。

    logits [T, J, 3]，target [T, J]（0、1、2 对应 −1、0、+1）。变化（±1）远少于不变，两类各自取平均再相加，
    免得变化被稀释；某一类不存在时那一项为 0。
    """
    ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
    still, moved = mask & (target == 1), mask & (target != 1)
    loss = ce[still].mean() if still.any() else ce.sum() * 0
    return loss + ce[moved].mean() if moved.any() else loss


def sample_loss(model, outputs, sample, cfg, margin=0, event_weight=0.0):
    """L_STG + L_CNT + L_P + λ·L_interval，对样本中 grad_from 之后的各步取平均；边界附近不计 STG、CNT。

    训练时另加（验证不加，保持可比）：margin > 0 时 CNT 只对 r ≤ 真实次数 + margin 的门槛计损失；
    event_weight > 0 时加 event_weight · event_loss。
    """
    logits = {name: torch.stack([out[name] for out in outputs]) for name in ("stage", "count", "progress")}
    labels = {name: sample[name][sample["grad_from"]:].to(model.special.device) for name in LABELS}
    stage_mask, count_mask = labels["stage_mask"], labels["count_mask"]
    if margin > 0:   # 远处的负门槛（如固定流程的"至少 4 次"）不计，免得它们把阈值推开、稀释重复任务的梯度
        thresholds = torch.arange(1, count_mask.shape[-1] + 1, device=count_mask.device)
        count_mask = count_mask & (thresholds <= labels["count"].sum(-1, keepdim=True) + margin)
    stage = F.cross_entropy(logits["stage"][stage_mask], labels["stage"][stage_mask]) \
        if stage_mask.any() else logits["stage"].sum() * 0
    # 计数：每个 (帧, 项) 对 r 求和，再对 (帧, 项) 取平均。直接对 J×R 个输出取平均时，大量容易的负例
    # （如固定流程的"至少 2 次"）会把"至少 1 次"这类关键判断的梯度稀释约 R 倍。
    if cfg.count_mode == "event":
        # 目标是"把回传的次数纠正到真值"所需的变化：无误差时就是完成事件，回传有误时是纠正。
        fed = torch.stack([out["fed"] for out in outputs])
        target = (labels["count"].sum(-1).long() - fed).clamp(-1, 1) + 1
        count = change_loss(logits["count"], target, labels["event_mask"])
    else:
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
    if cfg.count_mode == "event":   # 回传的次数加上预测变化的期望
        change = F.softmax(logits["count"].detach(), -1) @ torch.tensor([-1.0, 0.0, 1.0], device=fed.device)
        k = (fed + change).sum(-1)
    else:
        k = logits["count"].detach().sigmoid().sum((-2, -1))
    total = sample["total"]
    interval = (F.relu(k / total - p) + F.relu(p - (k + 1) / total)).mean()
    event = event_loss(logits["count"], labels["count"]) if event_weight > 0 and cfg.count_mode == "level" else count * 0
    loss = stage + count + progress + cfg.interval_weight * interval + event_weight * event
    return loss, {"stage": stage.item(), "count": count.item(), "progress": progress.item(),
                  "interval": interval.item(), "event": event.item()}
