# π0.5 / VLA-FAIL 集成与 4090A 数据测试入口

此分支将用户本地 `VLA-FAIL/Code` 的独立实现整合为 `packages/vla-fail`，并接入 OpenPI 的 **PyTorch π0.5**。方法依据 VLA-FAIL（arXiv:2606.21386v1）；这不是作者官方发布代码，也不是论文实验结果复现。不使用世界模型。

## 实现范围

| 模块 | 行为 |
|---|---|
| LLMD | 每个动作预测位置分别统计均值和完整协方差，正则化后用 Cholesky 求平方马氏距离，取所有位置最大值 |
| 固定先验特征提取 | `action_out_proj` 前的动作专家最终特征 `[H,F]`，固定噪声，`eval()`，无梯度 |
| ACC | 对齐 `previous[R:]` 和 `current[:H-R]`；按每个坐标轴的位置极差归一化，再做 EMA |
| Policy 接线 | 可选开启，动作输出附带 `vla_fail` JSON；按 episode 重置历史；支持仅 LLMD、仅 ACC、两者 OR |
| 离线工具 | 特征导出、流式统计、独立成功轨迹校准、CSV 分数导出 |

尚未实现大模型接管、JEV、语义蒸馏、新增 VLM 正则项。返回值是异常信号及介入建议，不是失败概率；LLMD 高分也可能来自成功的泛化。

## 最关键的接线约定

1. **纯噪声端点是 `time=1`。** 本仓库训练使用 `x_t = t*noise + (1-t)*actions`。特征提取不能直接照其他时间参数化填 `t=0`。
2. 固定噪声必须在统计、校准、测试、上线中一致；NPZ 保存完整噪声数组。真实动作采样仍使用原有采样噪声。在线复用视觉/语言 KV cache，额外执行一次动作专家前向；有计算开销。开启 LLMD 时使用 eager 诊断路径，不能假设与原 compiled sampler 延迟相同。
3. LLMD token 是 **动作专家输出投影前的 token**，不是 VLM 图像 patch token。修正了原有语义窗口将原始图像通道数误当图像 token 数的计数方式。
4. **ACC 输入必须是相同坐标系、相同单位的绝对末端 XYZ**。在 `Policy` 输出反归一化及动作转换之后取值。关节位置需要先做正运动学；连续笛卡尔增量可用 `delta_xyz_to_absolute`，相对初始位置的 offset 则需另一种转换。当前在线适配器只接收明确配置的 `absolute_xyz`，不会自动猜测 Piper 动作的含义。
5. `R` 是两次预测间**实际执行**的动作步数，需要 `0<R<H`。接管、重置、没有重叠时重置历史。首帧 ACC 未就绪和未校准阈值均返回 `null`，不会伪装成“安全”。
6. 单个 Policy/Monitor 只服务一个环境流。现有 WebSocket 服务共用 Policy，不支持多个客户端交错推理；多个环境应分别启动服务。每次请求带 `episode_id`；episode 变化或显式 reset 会清除 ACC 和语义窗口。

## 轻量测试环境

在仓库根目录，仅安装检测模块，不安装完整 VLA：

```bash
python -m venv .venv-vla-fail
.venv-vla-fail/bin/python -m pip install -e './packages/vla-fail[test]'
.venv-vla-fail/bin/python -m pytest packages/vla-fail/tests -q
```

Windows 将 Python 路径换成 `.venv-vla-fail/Scripts/python.exe`。无 Torch 时采样器测试会明确跳过；完整集成检查要求安装/使用兼容的 Torch。采样器测试使用小型替身网络执行真实采样器方法，覆盖时间端点、输出投影前特征、动作与随机数一致性、一次 prefix 编码、窗口边界及 Policy 输出变换，不需要 GPU。

服务器部署位置为 `/home/zhaobo/EVO-Intervention`；独立 `.venv-vla-fail` 用于模块测试。该环境不是完整 OpenPI 推理环境。完整模型依赖仍按仓库安装步骤配置（包括仓库自带的 Transformers 修改），不要覆盖现有 LeRobot 环境。

## A 上正式数据实验的前置条件

A 的数据位于 `/home/zhaobo/lerobot_datasets`，模型位于 `/home/zhaobo/models`。本次集成不选择某个任务、修改原始数据或启动完整数据实验。

正式测试前需要确定一个数据集及与之匹配的模型：

- 本仓库读取 OpenPI PyTorch `model.safetensors` 和 `assets` 归一化统计。A 上现有 LeRobot π0.5 权重/配置不能仅根据文件名认定兼容，需要核对权重键、相机、状态、动作定义及归一化。
- 原始 LeRobot 字段应转换成相应 OpenPI Policy 的输入字段；比如所选 policy 的图像、state、prompt。下面的提取器接收已经映射好的原始 `Policy.infer` 观测，不直接解码 LeRobot 视频/Parquet。
- 训练参考集、成功校准集、测试集按完整 episode 划分，不能随机拆帧。任务指令、相机处理、checkpoint、归一化配置要一致。
- 离线观测可以评估 LLMD。ACC 要在同一条轨迹上记录连续的**模型预测动作块**及实际 `R`；数据集的真实执行动作/专家动作不是这些预测块的替代品。离线回放得到的是 counterfactual 一致性，不等同于实际闭环策略失败率。

## 1. 导出 LLMD 特征

每帧保存一个无 pickle 的 NPZ，键和值对应所选 `Policy.infer` 输入；`prompt` 存为字符串标量。JSONL 清单每行一个 episode，观测路径相对清单所在目录：

```json
{"episode_id":"train-0001","observations":["frames/0001.npz","frames/0002.npz"]}
{"episode_id":"cal-0001","observations":["frames/0101.npz","frames/0102.npz"],"success":true}
```

实际训练/校准/测试应使用分开的清单与输出目录。示例命令中的配置和 checkpoint 需替换为已验证兼容的实际值：

```bash
python scripts/extract_vla_fail.py \
  --config pi05_libero --checkpoint /path/to/openpi_checkpoint \
  --manifest /path/to/train.jsonl --output artifacts/train --device cuda

python scripts/extract_vla_fail.py \
  --config pi05_libero --checkpoint /path/to/openpi_checkpoint \
  --manifest /path/to/calibration.jsonl --output artifacts/calibration \
  --fixed-noise artifacts/train/fixed_noise.npy --device cuda
```

输出 `episode_*.npz` 的格式：

| 键 | 形状/内容 |
|---|---|
| `features` | `[T,H,F]`，每次观测的动作专家特征 |
| `fixed_noise` | `[H,action_dim]` |
| `metadata` | JSON 字符串，包含 `feature_source`、`flow_time`、`config`、`checkpoint` |
| `episode_id` | 全局唯一字符串标量 |
| `success` | 可选布尔标量，校准时必须明确为 true |
| `positions` | ACC 回放额外需要 `[T,H,3]` 的预测绝对 XYZ；提取器不会虚构这些值 |
| `executed_steps` | ACC 回放额外需要 `[T]` 整数数组，第一项忽略，此后是实际 R |

提取器不会覆盖已有 episode 文件。checkpoint 元数据比较使用传入的原始字符串，跨机需保持相同标识/路径。它不是权重内容校验；不要在同一路径替换 checkpoint 后继续使用旧统计量。

## 2. 拟合、校准、导出分数

```bash
python -m vla_fail.cli fit artifacts/train/episode_*.npz --output artifacts/reference.npz

python -m vla_fail.cli calibrate artifacts/calibration/episode_*.npz \
  --stats artifacts/reference.npz --alpha 0.05 --output artifacts/calibrated

python -m vla_fail.cli score artifacts/test/episode_*.npz \
  --stats artifacts/calibrated/llmd.npz --output artifacts/test_scores.csv
```

测试集也使用上述提取器导出并复用训练 fixed_noise。Linux shell 展开通配符；Windows PowerShell 下请传实际文件列表。

校准使用每条成功轨迹的最大分数，按轨迹构造 split-conformal 阈值，拒绝重复 episode、与参考拟合集重叠的 episode、不成功标签及噪声/元数据不一致。统计量保存后不能混入校准数据重新拟合。

仅一个信号时，`alpha=0.05` 至少需要 19 条独立成功校准轨迹；LLMD+ACC 联合时分别分配 `alpha/2`，至少需要 39 条。这个误报控制依赖成功轨迹的可交换性，不保证 OOD 条件误报率、失败召回率或提前量。EMA 初始化为零，训练/校准/部署保持一致。

联合校准可加 `--acc-config /path/to/acc.json`（数据也须含 `positions`、`executed_steps`），输出 `llmd.npz`、`acc.json` 和 `calibration.json`。ACC 配置示例：

```json
{"action_representation":"absolute_xyz","position_indices":[0,1,2],"ema_alpha":0.9,"v_min":0.001}
```

此示例只适用于确实将绝对 XYZ 放在动作前三维的 policy；单位为米时 `v_min=0.001` 表示 1 mm 的归一化下限。配置中的 `threshold` 由校准命令写入。

CSV 包含 `episode_id,step,llmd,acc,acc_raw,need_intervention,decision_complete`，可直接绘制时间曲线，并结合真实失败时间计算失败召回、成功轨迹误报、介入提前量。没有阈值/历史的位置留空。

## 3. 推理服务开启监测

在完整 OpenPI 环境中运行，仅 LLMD：

```bash
python scripts/serve_policy.py --port 8000 \
  --vla-fail-stats artifacts/calibrated/llmd.npz \
  policy:checkpoint --policy.config pi05_libero --policy.dir /path/to/openpi_checkpoint
```

已确认绝对 XYZ 输出后，可以再加 `--vla-fail-acc-config artifacts/calibrated/acc.json`。不指定上述参数时监测关闭，原有调用方式继续有效。服务会校验统计量中的配置名、checkpoint 字符串、特征源、时间端点以及噪声形状。

请求观测增加一个控制字段：

```python
obs["vla_fail"] = {"episode_id": "trial-001", "executed_steps": 10}
result = policy.infer(obs)
print(result["vla_fail"])  # need_intervention / reasons / llmd / acc / unavailable
```

第一帧不需要 `executed_steps`；下一帧必须填写真实执行步数。接管返回时发送 `reset: True` 或本地调用 `policy.reset()`；默认 ActionChunkBroker 用完全部 H 步才重新请求，此时没有重叠，不能直接启用 ACC，需要配置执行较短的动作前缀并正确传递 R。异常后重新开始 episode，避免继续沿用不明确的执行历史。

## 性能和验证边界

LLMD 当前用 CPU float64 完整协方差；`H=50,F=1024` 时单份 `[H,F,F]` 约 400 MiB，离线拟合峰值更高，分解成本也较高。应先抽取少量样本测内存与延迟，再扩大统计；未做 GPU 马氏距离或降维优化。不能把该基线称为“零开销”或未经实测保证实时性。

已覆盖数学公式、对齐/EMA、校准与缺失状态、工件往返、HTTP、运行时重置及微型网络上的采样器接线。完整 π0.5 checkpoint、A 的真实数据检测有效性和 MuJoCo 闭环尚待后续实验。
