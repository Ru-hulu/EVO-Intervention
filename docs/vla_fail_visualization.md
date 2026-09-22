# LLMD / ACC 单轨迹可视化

处理流程分为三步：从原始 LeRobot v3 数据导出观测，离线运行 π0.5 取得内部特征与完整预测动作块，最后计算并绘制指标。所有脚本只做离线计算，不连接机械臂。

已完成的两条真实失败轨迹及实验设置见 [2026-09-22 运行记录](vla_fail_rollout_manual_results.md)。该次预览使用 B 上的 15k 模型，每 30 帧查询一次；下面的通用示例默认每 10 帧查询一次。

## 脚本和输出

- `scripts/prepare_lerobot_vla_fail.py`：按照 episode 元数据定位视频文件和时间偏移，对齐三路相机与状态，输出观测 NPZ。
- `scripts/replay_lerobot_vla_fail.py`：加载本地 LeRobot π0.5 权重和保存的前后处理器，严格检查权重键，提取固定噪声下 `t=1` 的动作专家投影前特征，并重新预测动作块。
- `python -m vla_fail.visualize`：计算 LLMD、左右臂 ACC，输出 `scores.png`、`scores.svg`、`scores.csv`、`summary.json`。提供观测 bundle 时还会输出 `frames.png`，展示起点、两个指标峰值和终点的三路画面。

可视化使用可选依赖：

```bash
python -m pip install -e './packages/vla-fail[viz]'
```

前两个脚本额外需要已配置好的 LeRobot π0.5、PyTorch、PyAV、PyArrow；回放 Piper 关节动作还需要 `piper_sdk`。仓库根环境的旧版 LeRobot 与新部署权重未必兼容，建议使用独立环境引用已验证的部署运行时，避免覆盖现有训练环境。

## 1. 导出一条或多条原始轨迹

```bash
python scripts/prepare_lerobot_vla_fail.py \
  --dataset /inspire/hdd/evomind/lerobot/rollout_autonomous_5bolts_manual_20260920_160555 \
  --episodes 0 1 --stride 10 --output artifacts/rollout_manual/bundles/failure
```

默认相机为 `observation.images.right_environment_1`、`observation.images.left_wrist`、`observation.images.right_wrist`。若示范集顶视图名为 `observation.images.top`，添加 `--base-camera observation.images.top`。视频解码会检查时间误差；图像使用部署运行时的 π0.5 resize/pad 函数缩放至 224×224，保存 float32，避免额外 JPEG 压缩。

`--stride 10` 在 30 FPS 数据上约每 0.33 秒重新查询一次策略。`--sample-count 80` 可替代固定间隔，在一个参考 episode 中均匀抽取 80 帧；仅用于参考特征统计。ACC 回放必须保持查询间隔小于动作预测块长度。

## 2. 运行模型并建立 LLMD 参考

`MODEL` 必须指向本次实际要评价的模型，不能只根据采集轨迹的名字推断。

```bash
MODEL=/path/to/local/lerobot_pi05_checkpoint
python scripts/replay_lerobot_vla_fail.py --checkpoint "$MODEL" \
  --reference artifacts/rollout_manual/bundles/reference/episode_*.npz \
  --episodes artifacts/rollout_manual/bundles/failure/episode_000000.npz \
             artifacts/rollout_manual/bundles/failure/episode_000001.npz \
  --output artifacts/rollout_manual/features

python -m vla_fail.cli fit artifacts/rollout_manual/features/reference/episode_*.npz \
  --output artifacts/rollout_manual/reference.npz
```

若 checkpoint 目录没有 tokenizer，添加 `--tokenizer /path/to/local/paligemma-tokenizer`；可用 `--reference-note` 写入参考集来源。跨机分批传输观测时，可用 `--wait-bundle-seconds 1800` 等待明确列出的 bundle 原子落盘，避免读取半个文件。此时应传完整文件名列表，不能让尚未出现的文件参与 shell 通配符展开。

第一次回放会用相同动作噪声对比原采样器，验证增加特征采集后动作一致。LLMD 固定先验噪声和实际动作噪声分开管理；权重加载失败或 token 非有限值会报错，不会退回随机模型生成图。

输出还包含完整预测关节块、左右臂绝对 XYZ、采样帧索引、时间戳、每次查询耗时和来源信息。关节角按数据格式以度输入，转换为弧度后传入 Piper SDK 的 FK，SDK 毫米输出转换为米。`--dh-offset 0/1` 指定 SDK 的 DH 偏移版本，默认 1；真实硬件参数应核对。左右臂各自在其固定基座坐标系内比较。

## 3. 分别绘制两条错误轨迹

```bash
python -m vla_fail.visualize artifacts/rollout_manual/features/failure/episode_000000.npz \
  --stats artifacts/rollout_manual/reference.npz \
  --bundle artifacts/rollout_manual/bundles/failure/episode_000000.npz \
  --output artifacts/rollout_manual/reports/episode_000000

python -m vla_fail.visualize artifacts/rollout_manual/features/failure/episode_000001.npz \
  --stats artifacts/rollout_manual/reference.npz \
  --bundle artifacts/rollout_manual/bundles/failure/episode_000001.npz \
  --output artifacts/rollout_manual/reports/episode_000001
```

图从上到下依次为 LLMD 最大平方马氏距离、所有动作位置的 `log10(1+distance)` 热图、左右臂 ACC（浅色原始值、深色 EMA）、重叠预测块的平均绝对 XYZ 误差（毫米）。CSV 保留每次查询数值，首帧 ACC 留空。

LLMD 阈值来自 `reference.npz`；没有校准时图上明确显示无阈值。ACC 可用 `--acc-threshold` 传入独立校准的阈值，不应根据这两条失败轨迹的峰值倒推。双臂分别计算论文的三维 ACC，`acc_max` 是本实现的双臂汇总扩展，不是论文原始单臂指标。

有人工标注的准确失败起点时可以传 `--failure-time 秒数`。`episode_success=failure` 仅是整条轨迹的终局标签，不能自动当作每帧标签，也不能把终止时刻当作失败开始。

## 本次数据的解释边界

`rollout_autonomous_5bolts_manual_20260920_160555` 有 17 条人工标记失败的轨迹。episode 0、1 的采集模型字段均为 `pi05_piperx_full558_h100_r13_step50000`，动作是 14 维双臂关节值，未记录完整预测动作块和内部 token。

因此 ACC 来自新生成的离线预测，重查询间隔由导出脚本的 `--stride` 决定。原采集 RTC 行为、原始采样噪声和原始重规划时间未被复现；回放 RTC 关闭。若使用其他模型，报告必须明确记录该模型，不能解读为原采集策略的检测结果。新预测没有在环境里执行，下一帧仍来自原策略采集的轨迹，因此 ACC 还可能受到新策略与记录轨迹不一致的影响。

完整 full558 训练参考目录当前不可访问。探索性 LLMD 参考可以选同任务示范数据，但必须记录具体来源，并承认相机视角、覆盖范围和模型训练分布可能有差异。参考集和失败测试集必须分开。没有独立成功校准集时，本次图只用于查看分数变化和高分画面，不据此声称检测准确率、误报率或介入提前量。

`summary.json` 记录参考样本数、特征维数及协方差正则化强度。当样本数小于等于特征维数时，未经正则化的协方差必定秩亏，高分大小可能对正则化和样本覆盖很敏感。应先扩大参考统计，再讨论正式阈值。

此回放适配器导出的特征来源标识为 `lerobot.pi05.action_out_proj.input`；它与 OpenPI 适配器的产物分开校验。不同 checkpoint、预处理或运行时的 LLMD 统计量不能混用，也不能直接把本次 LeRobot 统计量放入原 OpenPI 服务。
