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
    suites: str = ""                # ProPriMem suite filter: Count,Order,State; empty selects all
    resume: str = ""                # 从这个 checkpoint 继续训练（模型、优化器、调度器、随机数状态）
    init_from: str = ""             # 只载入这个 checkpoint 的模型参数，优化器、调度器、迭代号从头开始（如 stage 2）

    # 流式结构（proposal 3.3）
    window: int = 15                # 最近 N 帧，5 Hz 下 3 秒
    chunk_size: int = 24            # 一次并行计算的连续帧数；独立于可见窗口和反传段长，必须 > 0
    warmup_chunk_size: int = 0      # No-grad history chunk; 0 inherits chunk_size, independent of LR warmup
    train_chunk_size: int = 0       # Gradient chunk; 0 inherits chunk_size, retains gradients across segment
    image_max_width: int = 0       # Optional aspect-preserving image bounds, 0/0 keeps original images
    image_max_height: int = 0
    activation_checkpointing: bool = False  # Recompute each backbone layer during backward, including the prefix
    max_keyframes: int = 16         # 关键帧总数超过它时删随机加入的帧；真正的关键帧不设上限
    # none / label（扰动后的标注状态变化帧）/ model（按 δ 触发）。默认 none：不保留历史关键帧，长期动态状态交给 S；
    # label 是"给外部记忆"的消融（step400 及之前的 checkpoint 都按 label 训练，它们的 config 里记着）
    keyframes: str = "none"
    # 计数回传默认关闭：False 只把回传概率置零，保留 h_STEP 与回传 MLP；旧 checkpoint 按 True 载入（checkpoint_config）
    count_feedback: bool = False
    oracle_count_feedback: bool = False  # 推理上限：回传上一帧真值累积编码，首步全零；需 count_feedback=True
    progress_reads_count: bool = True   # 消融：False 屏蔽 P 对 CNT token 的直接注意力
    # 各 readout 是否看窗口内之前步的 readout（都仍看自己）。默认不沿 readout 传播历史结论；
    # 仍保留首帧、视觉窗口和短卷积状态。
    # CNT：之前的 CNT；STG：之前的 STG；P：之前的 STG/CNT/P。旧 checkpoint 的 CNT、P 按 True 载入（checkpoint_config）
    count_reads_history: bool = False
    stage_reads_history: bool = False
    progress_reads_history: bool = False
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
    segments_per_trajectory: int = 1  # Ordered non-overlapping segments sharing one prefix; detach between segments
    max_updates: int = 0           # Optional update cap in epoch mode; 0 uses the full epoch budget
    segment: int = 6                # 带梯度的步数，5 Hz 下 1.2 秒；24 GB 显存下 2B 最多约 12
    # 默认单阶段：teacher=False；无计数反馈和关键帧的 level 模式按 chunk 并行，否则逐步计算。
    # teacher=True 是可选的真值回传并行训练方式；是否再训练预测回传模式由实验决定。
    teacher: bool = False
    boundary: int = 1               # 切换、完成前后 ±w 步不计 STG、CNT 损失
    keyframe_jitter: int = 2        # 标注关键帧随机偏移 ±步数
    keyframe_drop: float = 0.2
    keyframe_extra: float = 0.02    # 每帧被随机加为关键帧的概率
    demo_jitter: int = 3            # 示范关键帧在完成时刻前后随机偏移的原始帧数
    drop_wrist: float = 0.3         # 示范只保留头部视角的概率
    # 打破"按时间计数"的捷径（训练时生效，0 为关闭）
    pause_prob: float = 0.2         # 插入停顿的概率：1–3 处，每处某一帧重复 5–20 步，标签不变
    speed_prob: float = 0.2         # 随机一段 2 倍速（隔帧删）或 0.5 倍速（每帧重复一次）的概率
    rewind_prob: float = 0.0        # 仅对预处理目录中确认 rewind_safe 的任务生效；标签跟着帧回退
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
    epochs: int = 0                 # >0: traverse all episodes each epoch, with repeat_boost oversampling
    iterations: int = 2000          # Legacy update budget, used only when epochs=0
    accumulate: int = 8

    # 日志、验证、保存
    log_every: int = 1
    eval_every: int = 100           # CLI 训练设 0 时不建验证集，用于只测显存 / 吞吐
    eval_episodes: int = 12         # 训练中定期验证用的 episode 数（均匀抽取）
    final_eval_episodes: int = 0    # 训练结束时验证的 episode 数，0 为验证集全部
    eval_at_start: bool = True      # 训练前先验证一次，作为基线
    press_eval: bool = False        # 全部 press_button 验证轨迹：原序 + 固定种子的两种不规则间隔，按第 k 次按压统计
    best_metric: str = "val/model/progress_mae"   # 选 best.pt 的指标，越小越好
    save_every: int = 500
    seed: int = 0
    device: str = "cuda"


# 改过名的配置项：旧 checkpoint、旧 YAML 里的旧名按新名读
RENAMED = {"p_reads_count": "progress_reads_count", "p_reads_history": "progress_reads_history"}


def renamed(values):
    return {RENAMED.get(key, key): value for key, value in values.items()}


def checkpoint_config(saved, **overrides):
    """checkpoint 里存的配置，加上命令行的覆盖（值为 None 的不覆盖）。

    计数回传、CNT 读历史 CNT、P 读历史 readout 原先默认开启，缺少这些字段的旧 checkpoint 按开启载入；旧名按新名读。
    """
    old = {"keyframes": "label", "count_feedback": True, "count_reads_history": True, "progress_reads_history": True}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    cfg = Config(**{**old, **renamed(saved), **renamed(overrides)})
    if cfg.oracle_count_feedback and not cfg.count_feedback:
        raise ValueError("真值回传需要 count_feedback=True（--count-feedback）")
    return cfg


def parse_config(argv=None, description=None):
    import yaml

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="YAML 配置文件")
    for field in fields(Config):
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if field.type is bool else field.type
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=kind, default=None)
    args = vars(parser.parse_args(argv))
    path = args.pop("config")
    values = renamed(yaml.safe_load(Path(path).read_text()) or {} if path else {})
    unknown = set(values) - {field.name for field in fields(Config)}
    if unknown:
        raise ValueError(f"{path} 中有未知的配置项：{sorted(unknown)}")
    values.update({name: value for name, value in args.items() if value is not None})
    return Config(**values)
