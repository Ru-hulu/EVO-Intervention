# ruff: noqa: RUF001, RUF002, RUF003
"""只预测 p 的流式模型的超参。取值优先级：命令行 > YAML 配置文件（--config）> 这里的默认值。"""

import argparse
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Config:
    model: str = ""                 # Qwen3.5 权重目录
    labels: str = ""                # 标签目录，这里只用划分、指令和帧号，不用子任务边界
    data: str = ""                  # 视频目录
    out: str = "outputs/progress_lite"
    suites: str = ""                # 逗号分隔，空为全部
    resume: str = ""                # 从这个 checkpoint 继续训练

    # 结构
    window: int = 5                 # full attention 看最近几帧，5 Hz 下 1 秒
    image_max_width: int = 320      # 画面只缩小不放大
    image_max_height: int = 240
    agibot_max_width: int = 224     # AgiBot 任务的画面上限，解码时就缩小，每帧 token 数约为其他三视角任务的一半
    agibot_max_height: int = 168
    progress_bins: int = 21
    head_hidden: int = 1024
    lora_rank: int = 16
    lora_alpha: float = 32
    # 新颖度写入门：patch 和上一帧同视角比较，1−cos 低于 low 不写入也不衰减 S，到 high 恢复原来的 β、α
    novelty_gate: bool = True
    novelty_low: float = 0.15
    novelty_high: float = 0.39

    # 时间增强
    pause_prob: float = 0.2         # 插入停顿的概率：1–3 处，每处某一帧重复 5–20 步

    # 反传
    segment: int = 32               # 整条轨迹切成这么长的段，全部带梯度；段之间切断梯度，5 Hz 下 6.4 秒
    train_chunk_size: int = 16      # 带梯度时一次并行的帧数，不截断段内梯度
    eval_chunk_size: int = 24       # 验证时一次并行的帧数
    activation_checkpointing: bool = False  # 打开后显存少得多，但明显更慢；段长超过 32 时要打开

    # 优化：三组学习率，线性预热后余弦衰减到 min_lr_ratio 倍
    head_lr: float = 1e-3           # [P] 和 MLP 头
    lr: float = 1e-4                # LoRA
    gate_lr: float = 1e-4           # GDN 的 in_proj_a/b、A_log、dt_bias
    warmup: int = 50
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    epochs: int = 1
    accumulate: int = 2             # 每次更新的轨迹数
    repeat_boost: float = 2.0       # 每个任务的样本数是最大任务轨迹数的几倍，固定流程任务不加倍

    # 验证、保存
    eval_every: int = 200
    eval_episodes: int = 24
    save_every: int = 200
    workers: int = 4
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
