# 全量轨迹 ACC 可视化后台任务

## 范围与输出

- HF：`MINT-SJTU/RW-RL-Dataset` 的 `piperx_insert_copper_screw` 子集，下载版本 `71790b3f47b2f4429c22b95b15e89d5b21ecd707`。558 条，1,255,729 帧。
- 本地失败数据：`rollout_autonomous_5bolts_manual_20260920_160555`。17 条，157,592 帧；原生标签为 failure。
- 合计 575 条，1,413,321 帧，30 FPS，约 13.09 小时。
- 所有轨迹完整覆盖，每秒推理一次，共 47,397 次预测。视频仍保留完整 30 FPS 时长；数值因果保持至下一次评分，无插值。
- 本任务只计算 ACC，不需要 LLMD 参考统计，不执行新的 LLMD 全量拟合。

最终输出根目录：4090B `/home/zhaobo/EVO-Intervention/artifacts/acc_all_20260923`。

| 文件／目录 | 用途 |
|---|---|
| `plan.json`、`run_manifest.json` | 冻结的数据、模型、参数和运行标识 |
| `status.json` | 当前阶段、已完成数量；失败时包含错误 |
| `predictions/` | 模型预测动作块和末端 XYZ；可以免推理重新评分 |
| `reports/` | 每条轨迹的 `scores.csv` 和 `summary.json` |
| `ranking.csv`、`ranking.json` | 全部 575 条轨迹的排序 |
| `selection.json` | 最低两条、最高两条的精确规则和分数 |
| `selected/` | 四条精选，使用相同纵轴范围 |
| `videos/hf_train/`、`videos/local_failure/` | 全部 575 条同步可视化视频；单视频纵轴自动调整 |
| `SELECTED_COMPLETE.json` | 四条精选已完成并校验 |
| `COMPLETE.json` | 全部视频已完成并校验 |

每个视频显示三个相机视角、左右臂原始 ACC、EMA ACC、动作块重叠位置误差（mm），以及随原始视频移动的时间线和当前数值。所有成品校验帧数与时长，特别使用 episode 的视频偏移，避免共用同一视频文件的失败轨迹从错误时刻开始。

## 模型与计算定义

统一模型是与失败采集记录一致的 `pi05_piperx_full558_h100_r13_step50000`。文件在从 A 传输至 B 后验证 SHA-256；正常动作采样直接调用模型，不加 LLMD 诊断分支。权重严格加载，FP32，RTC 关闭，固定动作随机种子规则 `20260922 + 原始帧编号`。

动作块长度以检查点配置为准，本次为 $H=50$。每秒一帧对应 $R=30$，两次预测有 20 个重叠动作。时间间隔由记录的控制帧确定，不使用推理耗时作为 $R$。Piper 关节角度转弧度后经 SDK FK（DH offset=1），从毫米转成米，左右臂分别计算绝对末端 XYZ。

$$v_{t,d}=\max(\max_i a^t_{i,d}-\min_i a^t_{i,d},0.01),$$

$$\hat S_t=\frac1{3(H-R)}\sum_{d=1}^{3}\sum_{i=1}^{H-R}\frac{|a^{t-1}_{R+i,d}-a^t_{i,d}|}{v_{t,d}},$$

$$S_t=0.9S_{t-1}+0.1\hat S_t.$$

$v_{\min}=0.01$ 米使用论文实机设置。第一帧无历史动作块，ACC 留空，不伪造为 0；之后 EMA 从 0 初始化。这项初始化是明确的实现选择，不能声称论文指定。未做成功校准，不画虚构的报警阈值。

本任务是对已录制观测的**离线反事实重放**：模型没有实际控制这次视频中的动作，ACC 反映该策略在这些观测上的预测一致性，不等于原采集运行中保存的动作块。

## 如何定义“最好两条、最坏两条”

每个评分时刻先取 $M_t=\max(S_{t,\mathrm{left}},S_{t,\mathrm{right}})$，再按整条轨迹有效评分的 **95 分位数**排序。分数相同时按中位数、episode ID 排序。

- 最好：全体 575 条中 P95 最低的两条，文件名明确写作 `ACC最低`。
- 最坏：P95 最高的两条，文件名明确写作 `ACC最高`。
- 不按单次尖峰的最大值排名；同时保存中位数、最大值供解释。
- 先评完所有轨迹才生成最终排名，发生异常不会把未完成的部分排名宣称成完整结论。

这是固定规则自动选取的极端案例，**不是任务成功率排名，也不是代表性抽样**。低 ACC 可能来自停滞或稳定地做错，高 ACC 可能来自正常纠偏。HF 示范没有原生成功标签，视频标注为“训练示范（无原生结果标签）”；失败源标签保留。

## 后台执行与续跑

1. A 上传输脚本用 rsync 断点续传检查点和两套原始数据至 B，完成校验后写 `STAGED.json`。
2. B 上批处理等待这份标记，检查 575 条完整输入，等待显存充足后加载模型一次，逐条预测和保存。
3. 预测文件采用原子写入。相同计划重启时跳过已有预测；计划不同则拒绝混用。
4. 完成评分后自动排序，先生成四条精选，再以两个 CPU 编码进程生成全部视频。
5. 本机交付脚本等待四条精选完成后取回四个视频和排行榜，将视频复制到桌面。它不调用模型，不需要助手持续监视。

重新运行同一批次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=/home/zhaobo/EVO-Intervention/artifacts/acc_all_20260923/runtime \
/home/zhaobo/EVO-Intervention/.venv-vla-fail/bin/python \
  /home/zhaobo/EVO-Intervention/artifacts/acc_all_20260923/runtime/scripts/batch_acc_videos.py \
  --plan /home/zhaobo/EVO-Intervention/artifacts/acc_all_20260923/plan.json \
  --output /home/zhaobo/EVO-Intervention/artifacts/acc_all_20260923
```

进程锁阻止同时启动第二个重复任务。完整代码快照随任务保存，不依赖后续工作区脚本变动。
