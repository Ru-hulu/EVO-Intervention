# FOREST 验证集 Dataset Viewer

独立复制 dataset-viewer 模板，仅为此次 5 条验证集前向增加标注 / 预测曲线。没有修改 dataset-viewer 或 cover-progress-viewer。

- URL: http://127.0.0.1:18789/ （SSH 转发到华为 127.0.0.1:8789）
- 云端结果及权重快照：`/cache/cvpzhr/evaluations/forest-s01-val5-20261106`
- 固定 5 条 val episode：4、54、81、99、132；按类别选择，未按预测结果筛选。
- 权重：启动时的 last.pt 快照，step 6600。GPU 0，batch 4。
- 正常因果历史缓存、5 Hz、包含末帧；无暂停增强。缓存与训练共用冻结视觉编码和 StreamForest 规则，无跨轨迹状态。
- 绿色：v3 下降标签；橙色：模型预测。预测点间只作显示插值，指标只算实际采样点。
- `predictions.jsonl` 保存采样帧号、真值与预测；`metrics.json` 保存同口径软标签 CE 及 MAE/RMSE（5 条子集，不是完整 val）。
- 小型媒体副本仅在华为云存放；不下载原始视频到本机。
- `plot_metrics.py` 的 loss 图上半是 train CE，下半是完整 val 的 MAE/RMSE，两者不是同一指标。

## 分类概率可视化

该独立 viewer 增加默认展开的 101-bin 概率柱状图。横轴为 progress 档位 0%–100%，纵轴为概率；点击柱子查看精确概率。视频播放、拖动时间轴、上一/下一推理帧均同步分布。5 Hz 观测之间保持上一观测的概率，不做插值，也不使用未来帧。

`progress/*.json` 新增 `bin_distribution`，保存 `bin_centers`、实际采样帧的 `probabilities` 和分布期望。`evaluate_five.py` 后续前向直接导出模型 `bin_probs`；旧结果由 `backfill_bins.py` 按各次原始权重快照、batch 大小、episode 顺序和因果缓存补算。每帧检查 101 维、非负、概率和为 1、分布期望与模型输出一致。

旧数据补算覆盖 step13224 的 Val10、Val5，以及 step6600 的 Val5。原始 progress JSON 备份为 `.before-bins.json`，原预测曲线和指标保持原值；面板标明补算分布期望与原预测的最大差异。补算结果、状态和日志位于华为 `/cache/cvpzhr/evaluations/forest-s01-bin-probabilities-20261106/`。
