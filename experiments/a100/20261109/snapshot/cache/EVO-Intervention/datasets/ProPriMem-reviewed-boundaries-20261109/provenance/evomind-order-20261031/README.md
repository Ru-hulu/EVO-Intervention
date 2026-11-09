# EvoMind LeRobot v3 → ProPriMem / Order

离线导出脚本，不连接机器人、不改原数据、不上传 Hugging Face，也不创建 PR。
格式已对照 `MINT-SJTU/ProPriMem` 的 `9ea7859fd65408152cab93a3fb4fb0ad0acf6dca` 版本。

`order_20261031.json` 明确列出清理后的 7 个来源，合计 237 条；不会扫描空批次、备份、已删除数据或其他新采集批次。新增“从左到右检查3个杯子底部”35 条已纳入。

| ID | 任务 | 轨迹 |
|---|---|---:|
| O08 | 按密码 | 30 |
| O09 | 按隐藏顺序操作按钮（合并清理后） | 36 |
| O10 | 按顺序敲击杯子 | 35 |
| O11 | 按顺序移动积木并归位 | 34 |
| O12 | 按顺序开关抽屉 | 35 |
| O13 | 从左到右翻转杯子 | 32 |
| O14 | 从左到右检查3个杯子底部 | 35 |

这些任务保留为新任务，不混入 HF 已有的 O01–O07；例如旧 O01 的动作顺序与此次抽屉任务不同。新增编号基于上述 HF 版本，后续合入前需要确认没有编号冲突。

## 运行

Python 3.10+，依赖 `av`、`numpy`、`pyarrow`。服务器现有环境已经具备依赖：

```bash
PY=/home/zhaobo/evomind-lerobot-dev/.venv/bin/python
WORK=/home/zhaobo/work/codex_active_vision/proprimem_order_export_20261031_v001
RAW=/home/zhaobo/work/codex_active_vision/memory_depend_collection_20261030_v001/data

# 只核对来源元数据、数量和异常短片段，不写导出目录。
"$PY" "$WORK/tools/export_order.py" --source-root "$RAW" --plan

# 每任务 1 条的试转：7 条 / 21 个视频，明确标为 partial。
"$PY" "$WORK/tools/export_order.py" --source-root "$RAW" \
  --output "$WORK/smoke" --limit-per-task 1

# 全量：237 条 / 711 个视频。输出目录必须不存在。
"$PY" "$WORK/tools/export_order.py" --source-root "$RAW" \
  --output "$WORK/full"
```

脚本默认读取旁边的清单，也可以通过 `--manifest` 指定经过确认的新清单。修改来源或数量前先结束对应任务的采集。脚本检查元数据帧数、数量、连续编号、三视角时间范围，并在完成后检查源元数据哈希及视频大小/修改时间。不会停止正在进行的其他任务。

## 输出与标注

```text
full/
  Order/
    08-enter_pin_sequence/
      metadata.json
      metadata.jsonl
      annotations/                 # 空目录：尚无人工阶段标注
      videos/main_view/episode_000000.mp4
      videos/left_view/episode_000000.mp4
      videos/right_view/episode_000000.mp4
    ...
    14-inspect_cup_bottoms_left_to_right/
  source-manifest.json
  provenance.json
  manifest.additions.json
  export-status.json
```

- 每个轨迹索引采用 HF 现有字段，视频路径相对于任务目录。主视角为环境相机 `right_environment_1`，左右视角对应左右腕相机。
- 按 LeRobot v3 的每段时间范围定位视频分片，精确切帧，保持分辨率和帧率。重新编码为**无损 H.264**，输出可能比原始视频大。逐输出完整解码，比对帧数、时间戳和原始片段的 YUV 像素哈希。
- 当前没有阶段标签：`has_annotation=false`，`annotation_path/task_prompt/num_stages/annotation_sha256=null`。真实动作指令在 `source_instruction`。不把完整动作时长或推测的阶段包装成真实标注；还不能用于要求阶段标签的监督训练。
- 导出为视频与索引格式，与现有 ProPriMem 对齐。机器人 action/state 的 Parquet 原样留在源目录，没有伪装成 ProPriMem 视频标注。
- 清单内各任务重新连续编号；`source_sample_id` 和 `provenance.json` 保留当前源数据集与 Episode 编号。源数据已经删除过错误轨迹，此处不会从备份恢复。
- 短轨迹提醒采用“少于本任务中位时长的 60%”作为复核启发式；**不会据此自动删数据或判断成功**。
- `manifest.additions.json` 按现有 80/20 规则列出新增任务的明确 train/val 样本列表及索引哈希。它只是**待合入片段**，不能直接覆盖 HF 现有 `manifest.json`，也没有修改原有冻结划分或发布版本。
- 输出完成只有 `export-status.json.status == "passed"` 才可用；试转还有 `partial=true`。失败可能留下不完整目录，原始数据不变，排查后换一个新输出目录重跑。完整索引只在一个任务全部片段通过校验后写入。

测试：在工具目录运行 `"$PY" -B -m unittest -v test_export_order`。测试覆盖非关键帧切割、第二视频分片从零重新计时、无标注字段、源数量变化、越界路径、截断视频及拒绝覆盖已有输出。
