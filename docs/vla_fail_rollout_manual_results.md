# 两条真实失败轨迹的 LLMD / ACC 可视化记录

日期：2026-09-22。脚本和绘图流程已在真实数据上跑通；本次是探索性离线回放，不是原采集策略的正式失效检测评估。

## 数据与模型

- 数据位于 4090A：`/inspire/hdd/evomind/lerobot/rollout_autonomous_5bolts_manual_20260920_160555`。
- 选择 episode 0、1；原始记录分别有 9,099、9,073 帧，30 FPS。两条均标记 `episode_success=failure`、`termination_reason=operator_failure`，没有逐时刻失败起点。
- 原采集模型：`pi05_piperx_full558_h100_r13_step50000`。
- **实际评估模型：4090B 的 `pi05_piperx_insert_copper_screw_bs64_15k_oss`。两者不是同一 checkpoint。** A 上 GPU 正在训练，跨机复制原模型速度约 1 MB/s，因此先使用 B 上已有的同任务模型出图。未停止 A 上训练。
- 实际评估在 B 上完成，使用 LeRobot、PyTorch 2.8.0+cu128、transformers 5.5.4、FP32。严格加载 checkpoint，关闭 RTC 和编译，不使用世界模型，不连接机器人。
- 数据中没有原始内部 token 或完整动作块。本次重新生成这些信号；新动作并未在环境里执行，后续观测仍来自原采集轨迹。因此 ACC 也可能反映评估策略与原轨迹的偏离，不能直接当作原策略在线不确定性。

## 实验设置

| 项目 | 设置 |
|---|---|
| 两条轨迹评估帧数 | 304、303 |
| 覆盖的记录时间 | 0–303 秒、0–302 秒 |
| 查询间隔 | 30 个记录帧，即 1 秒 |
| 预测动作块 | H=50，前后两次预测对齐后重叠 20 个动作 |
| LLMD 特征 | 动作专家 `action_out_proj` 输入，固定噪声、flow time=1，形状 50×1024 |
| LLMD 参考 | 8 条示范轨迹，每条均匀抽 80 帧，共 640 个观测 |
| 协方差 | 每个动作 token 单独估计完整协方差，加 1e-6 I |
| ACC | 左右臂分别对绝对 XYZ 计算，速度归一化后 EMA |
| FK | Piper SDK，关节度数转弧度，DH offset=1，毫米转米 |
| ACC 平滑与下限 | alpha=0.9，v_min=0.001 m |
| 阈值 | 未校准，LLMD/ACC 报警字段与介入决策保留为空 |

参考集来自 A 上 `piperx_insert_copper_screw_merged_393_cleaned_20260805_v1` 的 episode 0–7。逐条核对后，其状态数组、视频字节及时间偏移、任务字符串与本次 15k 模型训练集 `piperx_insert_copper_screw_cleaned_20260804_v2` 的同序号轨迹一致；记录见 `artifacts/rollout_manual/reference_source_comparison.json`。这是经核对的训练集子集，不是整个训练分布统计。

**640 个样本少于每个 token 的 1024 维特征，未正则化的协方差必定秩亏。** 目前的 LLMD 绝对数值会受正则化和参考覆盖影响；不得把它解释为失败概率。正式实验应扩大参考集并做正则化敏感性分析。

原先每 10 帧导出的完整观测仍保留在 A 的 `artifacts/rollout_manual/bundles/failure/`。本次从中每隔 3 个观测取一个，得到 1 秒间隔预览。alpha=0.9 在该间隔下的 EMA 半衰期约 6.6 秒，不能代表实时系统的告警延迟，也可能遗漏短暂异常。

## 结果

| 轨迹 | LLMD 峰值 | LLMD 峰值时间 | ACC EMA 峰值 | ACC 峰值时间 / 臂 |
|---|---:|---:|---:|---|
| episode 0 | 168,529.85 | 300 秒 | 4.9380 | 7 秒 / 右臂 |
| episode 1 | 316,445.71 | 290 秒 | 5.3778 | 40 秒 / 左臂 |

这里的 ACC 峰值取左右臂各自 EMA 分数的最大值，是双臂汇总方式。LLMD 使用平方马氏距离，不是开方后的距离。

episode 0 在中间较长一段时间内比较平稳，约 280 秒后 LLMD 与动作块差异再次明显波动。LLMD 中位数约 67,018，左右臂重叠 XYZ 的平均绝对误差中位数分别约 6.0、4.5 mm。

episode 1 全程有更多分数波动，LLMD 中位数约 120,923，左右臂重叠 XYZ 的平均绝对误差中位数分别约 14.4、18.6 mm。与 episode 0 相比，这个评估模型在第二条记录轨迹上给出了更多不一致的重规划。

两个信号的峰值时间明显不同，说明本次曲线捕捉到的变化并不相同。尚不能据此证明互补检测有效，更不能宣称某个峰值就是应该介入的时刻。整条失败标签无法确定失败起点；起步运动、场景变化和策略差异都需要结合视频进一步检查。

已查看两条轨迹的曲线和三相机抽帧图。抽帧图提供起点、ACC 峰值、LLMD 峰值、终点的对应画面，用于人工复核，不自动生成失败原因标签。

## 输出位置

本地、4090A、4090B 仓库下均保存两条轨迹的轻量报告：

```text
artifacts/rollout_manual/preview_15k/reports/
  episode_000000/
  episode_000001/
```

每个目录包含 `scores.png`、`scores.svg`、`scores.csv`、`summary.json`、`frames.png`。summary 保留实际模型、参考样本数、协方差正则强度、来源数据及查询间隔。

较大的参考统计 `reference.npz`、原始提取特征 `features/` 和推理日志保留在 B 的 `artifacts/rollout_manual/preview_15k/`。图像、原始数据、特征和模型权重未提交到 Git；本文件与处理代码提交到 `vla-fail-integration`。

## 在 B 上重新绘图

无需重新运行模型：

```bash
cd /home/zhaobo/EVO-Intervention
for ep in 000000 000001; do
  OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  .venv-vla-fail/bin/python -m vla_fail.visualize \
    artifacts/rollout_manual/preview_15k/features/failure/episode_$ep.npz \
    --stats artifacts/rollout_manual/preview_15k/reference.npz \
    --bundle artifacts/rollout_manual/bundles/failure_preview/episode_$ep.npz \
    --output artifacts/rollout_manual/preview_15k/reports/episode_$ep
done
```

从数据重新导出、推理与拟合参考的命令见 [可视化使用说明](vla_fail_visualization.md)。复现本次抽样可直接在导出时使用 `--stride 30`，参考集使用 `--episodes 0 1 2 3 4 5 6 7 --sample-count 80 --base-camera observation.images.top`。推理需要指定上述实际 15k checkpoint，并加 `--tokenizer /home/zhaobo/models/paligemma-tokenizer`；为避免覆盖，完整重跑应使用新的输出目录。

## 验证与下一步

- 本地测试 59 项通过，2 个需要额外模型运行时的模块跳过；B 上 64 项通过，包含实际 LeRobot 运行时下的适配器测试。
- Ruff 检查与格式检查通过。
- 真实模型回放首帧用同一噪声比较启用特征采集前后的动作，通过 `rtol=atol=1e-5` 的一致性断言。
- 没有成功校准轨迹、准确失败起点或原采集完整动作块，因此不报告检测准确率、误报率、提前量。

后续正式评估应使用原采集 checkpoint，补齐对应参考数据；增加独立成功轨迹校准阈值，并人工标注失败开始与可挽救时间范围。下一轮可在 A 空闲时用已保留的每 10 帧观测加密回放，并按采样间隔调整 EMA。
