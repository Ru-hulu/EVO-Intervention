# SOC 每任务两条的静止与回环缓存

只准备评测数据，不运行模型预测或训练。使用 A100 当前 State / Order / Count
manifest 的验证划分，固定 seed 42，每个任务抽取两条；已由用户排除的
`Order/O03/swap-blocks_episode_001342` 在抽样前排除。当前共 42 个任务、84 条轨迹
（State 22、Order 32、Count 30），抽样不依据 PT 是否已有。

沿用已确认的三个扰动种子 42、43、44，静止与回环分开，各 252 个版本，共 504 个版本。
同一个版本可以被不同模型共同读取。

- 输入采样沿用第一组的 5 FPS 规则：每个精确采样时刻取其后第一个原始帧，并保留末帧。
- 每个版本在两个不同内部位置插入扰动。位置范围从第 15 个观测到倒数第二个观测，
  保证每个位置均可接受 5–15 步、即 1–3 秒的均匀随机时长。
- 静止重复锚点帧 5–15 次，其 PT 完全不变。
- 回环先倒放 5–15 步，再等长正放回到锚点，额外时长 2–6 秒；PT 直接取对应原帧值。
- 三视角共享播放映射。现有缺失腕部视角仍使用原先的 RGB 黑色 64×64 占位规则；
  CSV 中该视角路径留空，不为损坏或短视角伪造补帧。
- 所有原始帧按原顺序保留；记忆必须按实际播放顺序更新，不能复用干净轨迹的历史状态。

新增 S12 / O15 / O18 中抽中的五条缺失 PT，另存补充标签。使用原 Qwen3.5-4B
视觉编码器、全部原始帧、batch 16、BF16/SDPA、4096–262144 像素、post-merger
tokens 的 FP32 均值、相邻 L2 差分、FP64 累计归一化。直接复用 Forest 原函数，
核对实际视觉权重 SHA。原始数据、已有 PT、阶段标注及训练/验证划分不改写。

远程目录：

- 准备审计：`/cache/cvpzhr/setup/soc-two-per-task-perturbations-20261114/`
- 最终缓存：`/cache/cvpzhr/datasets/soc-two-per-task-perturbations-5fps-20261114/`
- `selection.json`：当前划分、抽中轨迹、排除原因、全部输入文件 SHA。
- `inputs.json`：84 条原始 5 FPS 输入、原帧 PT、图片位置及指令。
- `images/`：按原帧索引保存的 PNG；已存在图片逐字节核验后硬链接，新增图片使用固定 PyAV 解码。
- `manifests/{static_only,rewind_only}-seed{42,43,44}.json`：六份完整播放清单。
- `playback/<条件>/<suite>/<task>/<sample>/frames.csv`：每一步的播放时间、原帧索引、原时间、PT、三视角路径。
- `verification.json`：独立逐点核验通过后才标 `ready`。

执行顺序：`prepare select`、CPU `prepare images`、仅空闲卡 `prepare labels`、
`prepare build`、独立 `verify`。入口模块前缀为
`experiments.soc_perturbations_20261114`。
使用原 `/cache/cvpzhr/envs/forest-labels/bin/python`，代码置于独立 setup/code 快照。
`labels` 拒绝占用中的 GPU，且要求 Recovery 无未启动队列；只加载冻结视觉编码器。
全部文件和清单冻结，重复执行不得覆盖不同内容。

这份缓存提供可直接用于评测的图片、播放顺序和真值，不预计算模型语言输出或扰动后的记忆。
任何预测输入应由图片及指令组成，不能把 CSV 的 PT 列传入模型。
