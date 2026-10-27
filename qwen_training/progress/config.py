# ruff: noqa: RUF001, RUF002, RUF003
"""训练与模型的超参。取值优先级：命令行 > YAML 配置文件（--config）> 这里的默认值。"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    model: str = ""                 # Qwen3.5 权重目录
    labels: str = ""                # labels.py 的输出目录
    data: str = ""                  # RMBench 的 data 目录
    out: str = "outputs/progress"
    tasks: str = ""                 # 逗号分隔，空为全部任务
    resume: str = ""                # 从这个 checkpoint 继续训练（模型、优化器、调度器、随机数状态）
    init_from: str = ""             # 只载入这个 checkpoint 的模型参数，优化器、调度器、迭代号从头开始（如 stage 2）

    # 流式结构（proposal 3.3）
    window: int = 15                # 最近 N 帧，5 Hz 下 3 秒
    max_keyframes: int = 16         # 关键帧总数超过它时删随机加入的帧；真正的关键帧不设上限
    keyframes: str = "label"        # none / label（扰动后的标注状态变化帧）/ model（按 δ 触发）
    delta: float = 0.3
    isolate_gdn: bool = True        # readout 在 GDN 层 β=0、α=1、不进短卷积

    # 头与 LoRA
    # 计数头：level 每帧输出累积次数 P(n_j ≥ r)；event 每帧输出第 j 项这一帧的次数变化 {−1, 0, +1}，
    # 次数在模型外累加，回传给下一步的 [CNT]（event_weight、count_margin 只用于 level）
    count_mode: str = "level"
    head_hidden: int = 1024
    progress_bins: int = 21
    lora_rank: int = 16
    lora_alpha: float = 32

    # 样本
    segment: int = 6                # 带梯度的步数，5 Hz 下 1.2 秒；24 GB 显存下 2B 最多约 12
    # 两阶段分开训练：stage 1 teacher=True，[CNT] 回传真值计数、整段并行；stage 2 teacher=False，回传自己的预测、
    # 逐步计算，用 init_from 从 stage 1 的 checkpoint 开始
    teacher: bool = False
    boundary: int = 1               # 切换、完成前后 ±w 步不计 STG、CNT 损失
    keyframe_jitter: int = 2        # 标注关键帧随机偏移 ±步数
    keyframe_drop: float = 0.2
    keyframe_extra: float = 0.02    # 每帧被随机加为关键帧的概率
    demo_jitter: int = 3            # 示范关键帧在完成时刻前后随机偏移的原始帧数
    drop_wrist: float = 0.3         # 示范只保留头部视角的概率
    # 打破"按时间计数"的捷径（训练时生效，0 为关闭）
    pause_prob: float = 0.0         # 插入停顿的概率：1–3 处，每处某一帧重复 5–20 步，标签不变
    speed_prob: float = 0.0         # 随机一段 2 倍速（隔帧删）或 0.5 倍速（每帧重复一次）的概率
    event_prob: float = 0.0         # 带梯度的段必须包含一个计数变化帧（含末帧）的概率
    repeat_boost: float = 0.0       # 0 为按 episode 均匀抽；>0 为每个任务等概率，计数会超过 1 的任务再乘这个倍数
    count_margin: int = 0           # >0 时训练只对 r ≤ 真实次数 + margin 的门槛计 CNT 损失
    count_balance: bool = False     # 按计数等级（各项最大计数 0/1/2/3/≥4）等概率抽带梯度的段，见 data.count_levels
    inject_prob: float = 0.0        # 回传给 [CNT] 的次数注入一处漏检或误检的概率（teacher 训练时生效），见 data.sample
    # 让 [CNT] 用上画面（0 / False 为关闭）
    event_weight: float = 0.0       # 训练时事件损失的权重（计数分数只在完成时跳变），见 loss.event_loss
    count_norm: bool = False        # 计数头输入 [h_CNT; h_STEP] 按特征维标准化（滑动统计量）
    count_norm_samples: int = 16    # 训练前汇总标准化统计量用的训练样本数（train.fit_count_norm）
    workers: int = 2                # 后台读 HDF5、解码图像的进程数，0 为在主进程读

    # 优化：三组学习率，线性预热后余弦衰减到 min_lr_ratio 倍
    head_lr: float = 1e-3           # 随机初始化的部分：readout / [STEP] embedding、MLP 头、计数阈值、计数回传 MLP
    lr: float = 1e-4                # LoRA
    gate_lr: float = 1e-5           # GDN 的 in_proj_a/b、A_log、dt_bias
    warmup: int = 50                # 预热的参数更新次数
    min_lr_ratio: float = 0.1
    gate_l2: float = 1e-3
    interval_weight: float = 0.1
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    iterations: int = 2000
    accumulate: int = 8

    # 日志、验证、保存
    log_every: int = 1
    eval_every: int = 100
    eval_episodes: int = 12         # 训练中定期验证用的 episode 数（均匀抽取）
    final_eval_episodes: int = 0    # 训练结束时验证的 episode 数，0 为验证集全部
    eval_at_start: bool = True      # 训练前先验证一次，作为基线
    best_metric: str = "val/model/progress_mae"   # 选 best.pt 的指标，越小越好
    save_every: int = 500
    seed: int = 0
    device: str = "cuda"


def parse_config(argv=None, description=None):
    import yaml

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="YAML 配置文件")
    for field in fields(Config):
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if field.type is bool else field.type
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=kind, default=None)
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    values = yaml.safe_load(Path(path).read_text()) or {} if path else {}
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"{path} 中有未知的配置项：{sorted(unknown)}")
    values.update({name: value for name, value in args.items() if value is not None})
    return Config(**values)
