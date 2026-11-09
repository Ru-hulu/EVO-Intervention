# FOREST 最新验证轨迹展示

本次补跑使用新增 12 条历史配对训练数据后的 FOREST best.pt（step14604），保持正常因果历史、5 Hz 预测和 101-bin 原始分类概率。固定轨迹为此前 IMPRINT full-attention 5 条、recurrent 5 条，以及脚本 recovery/失败各 1 条；不按预测好坏重新挑选。

`evaluate.py` 和 `plot.py` 迁入自既有 FOREST 验证工具，推理逻辑保持不变。`run.py` 在四张 GPU 上分配完整 episode，并在全部成功后合并、核验、更新展示入口。

云端代码：`/cache/robot-baselines/repos/memory-vla-RL/downstream/eval/forest_latest`。

本次输出：`/cache/cvpzhr/evaluations/forest-s01-history-pairs-val12-step14604-20261107`。

Viewer：`http://127.0.0.1:18789/?dataset=FOREST-S01-Latest`。

## 展示规则

按用户要求，今后使用一个固定的 `FOREST-S01-Latest` 入口展示最新模型结果，替换前一次展示，不在下拉框累积版本。只有在完整前向、概率检查、标注与边界一致性验证成功后才切换入口。原始权重、GT、子任务边界和历史实验结果留作追溯；替换的是展示版本。

本次归档隐藏旧的 Val5 step6600、Val5 step13224、Val10 step13224 三个入口，展示 12 条唯一轨迹。模型筛选仍分别对应两版 IMPRINT 各 5 条，脚本样本单独标记。既有的模型执行轨迹工具 18788 不改动。

## 执行与验收

当前 `run.py` 固定为这次 step14604 补跑，拒绝覆盖已存在的输出目录。后续模型应使用新的不可变输出目录及相应的检查点标识，继续发布到同一个 Latest 别名；不要把旧运行参数误用于新检查点。

检查 `status.json` 的 `phase=completed` 与 `published=true`；发布前状态不代表 viewer 已更新。`metrics.json`、`selection.json` 和代码快照保存在输出目录，`previous-viewer-config.json` 保存原展示配置。

12 条前向与 28 条完整验证是不同集合，不能把这次选定轨迹的 MAE 当作全量验证 MAE。

本次训练数据为 v4；展示的仍是同一组 12 条验证轨迹，没有将新增训练样本混入验证展示。旧 step13224 输出完整保留，只替换 Latest 指向。
