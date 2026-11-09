# ProPriMem 三套 suite 预处理

输入为 `ProPriMem/{Count,Order,State}/<task>/metadata.json`、`metadata.jsonl`、
`annotations/*.json` 和各视角 MP4。只读取当前 suite 内的相对路径，不依赖迁移前的 RMBench / RoboValue 目录。

```bash
python -m qwen_training.datasets.proprimem.prepare \
  --root /root/autodl-tmp/datasets/ProPriMem \
  --out /root/autodl-tmp/EVO-Intervention/outputs/proprimem/labels

python -m qwen_training.datasets.proprimem.validate \
  --root /root/autodl-tmp/datasets/ProPriMem \
  --out /root/autodl-tmp/evo-setup-20261026/suite-epoch-review-20261030/media-audit.json
```

`prepare` 输出逐轨迹训练标签、`_splits.json`、`_tasks.json` 和 `report.json`。
不会修改视频或源标注，也不会覆盖已有划分。

## 划分和示范

- 同一任务、相同步骤定义的轨迹组成一个兼容组；不同步骤定义分开，防止示范 STEP 下标错配。
- 有完整 0–49 编号的 RMBench 任务保留原划分：0–7 示范池、8 验证示范、9–44 训练、45–49 验证。
- 其他任务按固定 seed 打乱：约 10% 作示范池，1 条作验证示范，约 10% 作验证，其余训练。
- 四部分轨迹互斥，至少需要 4 条兼容轨迹。训练示范来自示范池，验证示范固定，不使用验证轨迹作前缀。
- 缺标注、无法解析或计数超过 MAX_COUNT=10 的轨迹写入排除清单；不会补造标签。
- 输入指令采用上游 source_instruction，缺失时用任务通用名称，不把标注中本条轨迹的完成次数作为额外输入。

## 标签语义

`policies.py` 为全部 31 个任务声明标注解释方式与倒放许可。
正在执行的区间和累计已完成状态分别解析，完成时刻不会混用；LIBERO 的最终完成停留不算新动作。
RMBench 使用已有手写步骤，避免把积木位置等记忆答案写进 STEP。
其他任务从明确的完成边界构造实例，重复动作的示范只取每项第一次完成的画面。

P 是按标注实例归一化并在实例内插值的派生监督，不是官方物理进度真值。
AgiBot 的“源动作段”标签保留 `label_unit=source_action_segment`：一段可能包含多次物理动作，
不能将这种计数称为精确拍手、敲击等物理次数。O02 的现有边界只监督按键次数，不足以监督具体数字键身份。
本脚本只检查标注结构和一致性，不证明人工子任务边界在画面上正确。

## 时间增强许可

默认暂停、变速各 0.2，倒放 0。所有 31 个任务当前明确 `rewind_safe=False`；
只有人工确认整条任务的倒放和 CNT/STG/P 回退语义一致后，才可在本目录修改许可。
配置概率大于 0 不能绕过任务许可。新任务没有默认授权。

## 训练和验证入口

使用 `qwen_training/progress/configs/autodl_proprimem_4b.yaml`。
`suites: Count,Order,State` 可选择一个或多个 suite；`tasks` 仍可选择完整的兼容组路径。
共享 loader 兼容旧 RMBench 标签和新 `_tasks.json`，能读取两视角或三视角 MP4。

epoch 模式每个 epoch 遍历全部训练轨迹，每次访问抽一个随机 segment。
`repeat_boost>0` 保留任务均衡和重复任务过采样；增加的访问尽量均匀分配到各轨迹。
例如原来三个任务各 36 条、repeat_boost=2 时每 epoch 有 144 次采样，accumulate=8 时更新 18 次。
新三套 suite 的每 epoch 样本数按实际划分重新计算，启动时打印；不能沿用 144。
末尾不足 accumulate 条仍更新，按实际条数归一化，不丢轨迹、不跨 epoch 拼批。

`epochs>0` 时总更新数为 epochs × ceil(每 epoch 样本数 / accumulate)，用于学习率衰减。
`warmup`、`eval_every`、`save_every`、`log_every` 仍按参数更新次数计。
日志和 checkpoint 同时保留 global iteration、epoch 和 epoch_step。
新 epoch checkpoint 可在 epoch 中间恢复，且不受 worker 预取影响；
旧 iteration checkpoint 或改变轨迹/采样权重时，请用 init_from 开始新的 epoch 计划。
`epochs=0` 保留旧 iterations 模式。全局 count_balance 与逐轨迹覆盖的 epoch 定义冲突，不能同时启用。

`segment` 决定连续反传的帧数；`train_chunk_size` 只决定一次并行处理多少帧，不在块间截断梯度。
`warmup_chunk_size` 控制无梯度历史前向的粒度。长示范前缀可启用 `activation_checkpointing`，
逐层重算主干激活来换取显存，保留前缀和跨块缓存的梯度，评估及无梯度热身不重算。

## 验证覆盖

`validate` 对索引中的每个视角文件检查完整 SHA256、文件大小、FPS、帧数，并解码首/中/末 RGB 帧。
它是“全量文件、抽样帧”检查，不是逐帧完整解码，也不是全量模型推理。

## 同轨迹多段训练

`segments_per_trajectory: 4` 时，增强后的轨迹按时间分成四个区间，每区间随机抽一个
不超过 `segment` 帧的连续段；保留 `event_prob` 的边界采样偏好。短轨迹缩短各段，
四段不重叠。一次访问只抽一次示范、只构造一次前缀，按时间推进历史和四个训练段。
每段反传后 detach 全部 KV、GDN S、卷积状态和 STEP 表示；后续段保留记忆值，
不向前段或前缀反传。只有首段能沿前缀路径反传。

`accumulate` 仍按 segment 计数：4 段/轨迹、`accumulate: 8` 即每更新访问两条轨迹。
epoch 仍按加权轨迹访问定义，当前 4374 次访问对应 2187 次更新。
`max_updates: 2735` 将总更新数封顶为 2735，学习率调度也按该预算计算，约运行 1.25 个 epoch。
采样语义变化后用 `init_from` 载入旧权重，新运行保存后可用 `resume` 精确继续。
新配置输出到 `proprimem_4b_seg24_multi4_20261030`，保持原运行文件。

图像可用 `image_max_width` / `image_max_height` 在编码前等比缩小；0/0 保留原图。
当前 AutoDL 配置为320×240上限、segment32、关闭激活重计算，`drop_wrist: 0` 保留全部原有视角，
原视频不修改，训练、验证与专家前缀共用同一缩图规则。处理器仍按patch网格对齐尺寸。
从 `proprimem_4b_seg24_multi4_20261030/step1.pt` 恢复优化器和调度，2735步预算包含该已完成步。
