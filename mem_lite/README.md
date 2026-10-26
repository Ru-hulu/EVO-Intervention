# mem-lite：事件记忆进度预测

当前已实现 M₁训练样本构造及训练入口、事件编码器与累计事件队列、M₂历史压缩器、独立冻结的状态文本解码器、M₂数据构造/加载和第二阶段联合训练入口，以及 M₃进度预测模型与三视角输入接口。
事件编码器只输入视觉表征；专家 tokens 只保留在 M₁接口中，暂不接入。
M₃已补充预构造样本的加载接口、冻结历史模块及第三阶段训练入口代码。
M₁/M₂支持批量数据构建、共同的视频级训练/验证划分，以及由训练入口直接读取的统一清单。
已补充M₁切分JSON与M₂状态文本的独立生成评测入口，默认读取同一清单的验证集。
实际训练先完成M₁、M₂；第三阶段历史构造方式仍在讨论。逐帧PT采用下述事件等权规则，尚未构造完整M₃样本或启动训练。

## 文件组织

```text
mem_lite/
  configs/m1_data.json        抽帧、静止过滤、随机种子与各类样本数
  configs/m1_train.json       M₁训练配置，模型与样本路径需自行填写
  configs/event_encoder.json  事件压缩：32 tokens、4层及内部维度配置
  configs/m2_compressor.json  M₂内部维度配置，query数固定128、层数固定4
  configs/m2_data.json        M₂抽帧、静止过滤、每视频扰动次数和随机种子
  configs/stage2_train.json   第二阶段数据路径、模型配置、双分支学习率与损失权重
  configs/m3_model.json       M₃历史输入维度、101个进度取值点与LoRA配置
  configs/stage3_train.json   第三阶段数据路径、学习率、批次及检查点配置
  datasets/video.py          视频时间轴、固定候选抽帧、静止过滤、边界补帧
  datasets/m1_samples.py     四类扰动、队列切片、自回归标签与样本读取
  datasets/prepare_m1.py     单条视频的数据构造入口
  datasets/preview.py        样本检查页面与观测拼图
  datasets/m1_dataset.py     Dataset、原生多图拼接与答案标签掩码
  datasets/m2_samples.py     邻居约束扰动、同一组事件的全部前缀及后继状态标签
  datasets/prepare_m2.py     单视频M₂数据构造入口，保存图片与队列清单
  datasets/dataset_manifest.py 视频级划分、每视频随机种子、统一清单与相对路径读取
  datasets/prepare_dataset.py 批量构造M₁/M₂，支持先生成计划和中断后续建
  datasets/m2_dataset.py     M₂帧去重/批次组织、联合M₁样本的Dataset与Collator
  datasets/m3_inputs.py      M₃同一时刻三视角、任务指令及可选128个历史tokens
  datasets/m3_dataset.py     预构造三视角样本、历史缓存读取与PT监督隔离
  datasets/prepare_progress.py 原视频逐帧视觉delta、事件等权PT与来源记录
  datasets/prepare_progress_batch.py 批量处理已标注主视角视频，复用冻结ViT
  models/m1_segmenter.py     Qwen3.5-0.8B 语言层 LoRA 与自回归切分损失
  models/event_encoder.py    逐帧视觉表征 → 4层交叉注意力与FFN → 32个事件tokens
  models/event_queue.py      单视频事件累计、前缀读取与事件序号位置编码
  models/m2_compressor.py    累计事件队列 → 4层交叉注意力与FFN → 128个历史tokens
  models/state_decoder.py    历史投影 → 独立冻结的0.8B语言模型 → 状态文本损失
  models/stage2_model.py     M₁切分监督与记忆文本监督的联合前向
  models/m3_predictor.py     M₃原生ViT、语言LoRA、历史投影和连续进度分箱预测
  models/frozen_history.py   从第二阶段加载冻结记忆模块，供后续离线构造历史
  train_m1.py                单卡 / torchrun 训练、梯度累积与断点恢复
  train_stage2.py            第二阶段单卡 / torchrun 训练与完整断点恢复
  train_stage3.py            第三阶段单卡 / torchrun 训练入口和断点恢复
  evaluate_m1.py             M₁自回归切分JSON、有效性判断、边界误差与IoU评测
  evaluate_m2.py             M₂历史解码、状态文本/次数/子句顺序的生成评测
  evaluation/metrics.py     纯离线计分与分组汇总，不从loss推算准确率
  evaluation/runtime.py     评测参数、来源记录与预测/错误样本保存
  tests/test_m1_data.py      时钟、覆盖率、边界与端到端数据测试
  tests/test_m1_training.py  离线接口、掩码、LoRA 范围与检查点检查
  tests/test_event_memory.py  事件压缩、顺序、变长输入与队列梯度检查
  tests/test_m2_compressor.py  固定历史长度、全部事件读取及跨模块梯度检查
  tests/test_state_decoder.py  独立语言权重加载、答案掩码与文本损失的跨模块梯度检查
  tests/test_m2_data.py        邻居扰动、标签配对、图像去重与确定性M₁样本配对
  tests/test_dataset_build.py 批量构建、跨集合隔离、目录迁移与失败续建
  tests/test_stage2_training.py 联合前向、梯度范围、双损失归一化及训练状态保存恢复
  tests/test_m3_predictor.py  三视角/历史插入、RoPE、软标签、混合批次与冻结范围
  tests/test_m3_data.py       三视角顺序、历史因果时间、标签隔离与数据签名
  tests/test_stage3_training.py 冻结历史、检查点、梯度归一化与DDP可选历史分支
  tests/test_progress_labels.py 事件等权、完成帧锚点、静止帧和冻结视觉提取检查
  tests/test_evaluation_metrics.py 失败样本分母、稀疏边界映射及文本指标口径
  tests/test_generation_evaluation.py 微型原生模型生成、缓存隔离和两个评测入口
```

数据构造依赖仅为 PyAV、NumPy、Pillow，不需要安装 GPU 训练环境。

```bash
python -m pip install -r mem_lite/requirements-data.txt
python -m mem_lite.datasets.prepare_m1 \
  --annotation /path/task/annotations/episode.annotations.json \
  --output results/mem_lite/episode
```

默认从 `annotations/` 上级目录寻找 JSON 的 `video_file`，也可通过 `--video` 指定主视角视频。
`--main-view` 记录视角名称；M₁只读取主视角。另两路当前观测留给后续 M₃。
`--expert-tokens` 可记录已有专家 tokens 文件的绝对路径；没有提供时为 `null`，不生成虚假参考。
输出目录必须是新目录。换配置重做时使用另一个输出目录，保留样本可比性。

## M₁/M₂批量构建与统一清单

批量入口只读取`任务目录/annotations/*.annotations.json`及其主视角视频，不把未标注视频或腕部视角重复加入。
先以完整视频为单位、按任务分别划分训练/验证集，再生成扰动样本；M₁和M₂共用同一份划分。
两阶段的抽帧、静止过滤、边界补帧和扰动规则与单视频入口一致，不需要PT文件或模型权重。

默认验证比例为20%、划分种子42，可通过参数修改。每个任务的验证视频数按四舍五入取整；
当比例大于0且某任务有至少两条视频时，保证训练/验证各至少一条。单视频任务只进入训练集；比例为0时全部进入训练集。
由于按任务取整，实际总比例不一定恰好等于20%。这属于任务内视频划分，不是未见任务的泛化划分。

以下是数据构建命令，不会启动训练。可以先只生成划分计划：

```bash
python -m mem_lite.datasets.prepare_dataset \
  --dataset-root /path/MemLong \
  --output results/mem_lite/dataset_v1 \
  --validation-fraction 0.2 --split-seed 42 \
  --plan-only
```

检查`manifest.json`中的视频归属后，沿用该计划生成数据；省略`--plan-only`则直接完成两步：

```bash
python -m mem_lite.datasets.prepare_dataset \
  --manifest results/mem_lite/dataset_v1/manifest.json
```

输出按阶段、任务和视频组织：

```text
dataset_v1/
  manifest.json
  m1/<任务名>/<视频名>/samples.jsonl、frames/、summary.json、preview.html、...
  m2/<任务名>/<视频名>/samples.jsonl、frames/、summary.json、...
```

清单记录视频的train/validation归属、标注摘要、两阶段采样配置和每视频种子，
并汇总每个集合的M₁正负样本数、空队列跳过数、M₂队列数和文本监督前缀数。
扰动种子由阶段配置seed、阶段名和任务/视频标识确定，与根目录及构建顺序无关；不会让每条视频共用同一随机序列。
M₁/M₂分别保留各自的帧目录和样本专属边界，不把对方的扰动边界混入观测队列。

中断或某条视频失败时，清单记录错误并停止，不能将部分产物作为完整训练集读取。
使用同一个`--manifest`再次运行可续建：复用已完成的阶段目录，仅清理和重建该批次未完成的`.partial`目录。
标注或采样配置改变时需新建批次，续建不重新抽取训练/验证视频。
源数据迁移后可以同时传入`--dataset-root /new/MemLong`；构建完成后，样本路径相对清单解析，整个输出目录可迁移。

两个训练入口新增`--data-manifest`，自动只选取其中的train视频，不需要手写全部samples路径：

```bash
# 后续训练时使用；数据构建不会执行这些命令。
python -m mem_lite.train_m1 \
  --model-path /path/Qwen3.5-0.8B \
  --data-manifest results/mem_lite/dataset_v1/manifest.json

python -m mem_lite.train_stage2 \
  --model-path /path/Qwen3.5-0.8B \
  --m1-checkpoint /path/m1/epoch-01.pt \
  --data-manifest results/mem_lite/dataset_v1/manifest.json
```

也可在训练配置的`data_manifest`字段中指定清单。不能同时在命令行指定清单和独立sample-files。
未指定清单时，原来的单视频/显式路径接口仍可使用；续训默认使用检查点内已解析的样本列表。
独立评测入口通过`load_sample_files(manifest_path, "m1"或"m2", split="validation")`取得验证数据。
本入口不进行模型验证或推理评测，也没有独立测试集划分。

## 构造顺序

1. 读取零基、左闭右开 `[start_frame,end_frame)` 标注。`state_description` 不作为 M₁模型输入。
2. 以整条视频起点为固定时钟原点，每0.1秒选择当时最新的一帧。25 FPS源的帧间隔为2、3交替；记录实际帧号和 PTS。候选被过滤后不重置时钟，不重复低帧率源帧。
3. 补入所有原始区间的首帧 `s` 和末帧 `e-1`，这些帧免静止过滤。普通候选与上一张保留帧比较。灰度缩图、轻度模糊后，统计差值超过 `pixel_delta` 的像素比例；不足 `min_changed_fraction` 则过滤。默认12灰度级和0.2%为初始参数，需结合观测拼图检查。
4. 在原视频帧轴上进行四类随机扰动。比例从包含两端的2%网格采样，按原事件长度计算；整数帧四舍五入后裁到视频范围。严格满足覆盖率 `>80%` 才有效，恰好80%无效。`samples_per_case` 默认每类8次，因此每个原区间共32条样本；短区间取整可能产生重复样本，保留其抽样权重。
5. 从公共观测队列截取输入窗口。有效样本额外补入**自身目标区间**的 `s`、`e-1`，无论它们是否静止。不会把一条样本的扰动边界添加给其他样本。排序、去重并从0重新编号。
6. 有效样本输出 `{"valid": 1, "start_index": ..., "end_index": ...}`。末尾索引不包含在片段内，可等于队列长度。无效样本只输出 `{"valid": 0}`。无观测的样本跳过并记录，不修改有效性标签。

各类范围：case1 左0–40%、右0–10%；case2 两侧0–20%；case3 左0–10%、右0–40%；case4 两侧0–20%。
随机性仅由配置种子控制，不使用模型状态。输入窗口、覆盖率均以原视频帧轴计算，不以过滤后帧数计算。

## 输出与模型接口

所有产物集中在指定目录：

```text
frames/              原尺寸JPEG；公共观测与各样本补帧共用图片文件
samples.jsonl        每行一个输入、标签和构造元数据
annotation.json      原始标注副本
observations.json    每个候选/原始边界帧的保留决定和帧差分数
summary.json         实际源信息、完整配置、各类有效/无效数量
skipped.json         被跳过的空队列样本
observations.png     公共观测拼图
preview.html         中文检查页面；每类展示有效/无效样例
```

`samples.jsonl` 的字段：

- `inputs`：任务指令、专家 tokens 路径、主视角名和有序 `frames`。每帧保存队列编号、原视频帧号、实际秒数以及相对输出目录的图像路径。
- `target` / `target_text`：结构化切分结果与自回归训练答案。
- `metadata`：case、原事件、扰动窗口、覆盖率等审计信息；不输入模型。

`read_training_sample(sample, output_dir)` 返回 PIL 图像列表、任务与队列位置提示、专家路径和目标文本，供 Dataset 调用。
专家路径仅作为预留元数据；当前提示不提及专家，Collator 不读取该文件，不插入占位专家 tokens。
原始帧号仅供回溯；提示使用队列序号与队列内实际相对时间，未包含标注状态或真实切分边界。

## M₁训练代码

训练流程直接使用已构造的样本，不在线修改区间、抽帧或随机扰动，也不自动划分验证集。

- **输入**：任务指令、带队列序号和相对时间的主视角图像。多条样本可以有不同队列长度。
- **输出**：原语言头自回归生成切分 JSON。有效样本包含边界，无效样本只包含 `valid: 0`。
- **可训练部分**：语言 Transformer 层的 Linear 上插入 LoRA，包括全注意力、线性注意力和 MLP 投影。基座、ViT、视觉合并层、词嵌入和原语言头均冻结；LoRA 参数保留 FP32，CUDA 前向使用 BF16。
- **监督**：JSON 答案及结束符的 token CE；图像、任务提示和 padding 标签均为 `-100`。仅对答案位置计算词表 logits。梯度累积按全部 rank 的答案 token 总数归一化，处理每轮末尾不满累积步数的批次。
- **长度**：超过 `max_sequence_length` 时明确报错，不截断图像队列，以免破坏边界索引。可调整 `max_pixels` 或长度上限。
- **专家接口**：`M1Segmenter.forward(batch, expert_tokens=None)`，当前传入非空专家 tokens 会报未接入；不会静默忽略或伪造专家输入。

以下仅为后续远程环境的运行示例，不是本次已执行的训练。安装依赖时根据 GPU 环境选择 PyTorch 的 CUDA wheel：

```bash
python -m pip install -r mem_lite/requirements-train.txt
python -m mem_lite.train_m1 \
  --model-path /path/to/Qwen3.5-0.8B \
  --sample-files /path/to/prepared/episode_a /path/to/prepared/episode_b/samples.jsonl \
  --output-dir outputs/mem_lite/m1
```

多卡使用同一入口：

```bash
torchrun --standalone --nproc_per_node=8 -m mem_lite.train_m1 \
  --config mem_lite/configs/m1_train.json \
  --model-path /path/to/Qwen3.5-0.8B \
  --sample-files /path/to/prepared/episode_a \
  --output-dir outputs/mem_lite/m1
```

`batch_size` 是每卡微批大小；完整优化步的样本数为 `batch_size × GPU数量 × gradient_accumulation_steps`，每轮最后一步可能更少。
默认 batch、LoRA、像素数等是开发初值，未经过吞吐或显存调优。多卡按 `DistributedSampler` 等长分片，样本数不能整除卡数时会补齐少量重复样本。
模型和 processor 只读取本地目录，不自动下载权重。

输出集中于 `output_dir`：`config.json`、`train.jsonl`、周期覆盖的 `last.pt`，以及每轮的 `epoch-XX.pt`。
检查点保存 LoRA、优化器、下一个 batch 位置、优化步数、各 rank 随机状态，不重复保存冻结基座。

```bash
torchrun --standalone --nproc_per_node=8 -m mem_lite.train_m1 \
  --resume outputs/mem_lite/m1/last.pt --epochs 4
```

恢复时使用检查点内配置，`--config` 不覆盖它；可用显式命令行参数调整终止轮数/总步数和输出目录。
`--max-steps` 指累计目标步数。续训须保持相同 GPU 数量、样本内容及 batch/累积配置；迁移路径可通过 `--model-path`、`--sample-files` 指向同一份模型与样本的新位置。
训练入口仅支持 CUDA BF16。正式训练与多卡运行仍需联调；不把接口检查视为训练已验证。

## 事件编码器与事件队列

当前已实现以下模块接口并接入第二阶段训练入口，在线流式调度尚未实现：

```text
冻结的共享 ViT → 每帧空间 tokens
                     ↓ 按有效 [start_index, end_index) 切片
               事件内全部视觉 tokens + 帧序号位置编码
                     ↓
               32个可学习query
                     ↓ 4层 Cross-Attention + FFN
               32个压缩后的事件 tokens
                     ↓ 入队
               全部事件 / 指定前缀 + 事件序号位置编码
                     ↓
               M₂：128个query，4层 Cross-Attention + FFN
                     ↓
               固定128个历史 tokens
```

### 视觉表征与压缩

`M1Segmenter.extract_frame_features(pixel_values, image_grid_thw)` 使用同一个冻结 ViT，返回按图像顺序排列的张量列表。
每个张量为 `[该帧空间token数, visual_feature_dim]`，取 Qwen 原生视觉合并层之后、进入语言模型之前的表征，保留空间 tokens，不做每帧平均池化。
该方法只执行视觉编码，可将结果复用于多个事件；现有 M₁训练前向仍走原生图文路径，本次未改成与事件分支共用一次前向计算的联合训练入口。

`EventEncoder(events)` 接收嵌套列表：`events[事件索引][帧索引]` 是一帧的视觉表征。
不同事件可以有不同帧数，每帧也可以有不同 token 数。内部进行 padding 和注意力遮罩，不让补齐位置参与压缩。
输入维度由 ViT 读取，经可训练线性层投影到内部维度；默认内部维度512、8个注意力头、FFN扩展4倍、dropout为0，均可通过配置修改。这些是开发初值，未做效果或效率调优。

32个可学习 query 是所有事件共用的初始查询。四层独立参数的模块依次更新 query，每层以更新前的 query 为 Q、同一份事件视觉表征为 K/V；交叉注意力和 FFN 都使用 Pre-LayerNorm 与残差连接，最后做输出 LayerNorm。没有增加任务指令、专家输入或额外自注意力层。
返回形状为 `[事件批大小, 32, hidden_dim]`。

帧位置采用正余弦编码，使用事件内保留帧的顺序 `0,1,...`，同一帧的空间 tokens 共享帧位置。
它表示事件内先后顺序，不编码原视频绝对帧号或实际时间间隔；原 ViT 已编码的空间信息仍保留。

### 切片与入队接口

`encode_event_intervals(encoder, frame_features, intervals)` 使用观测队列的零基、左闭右开边界，返回对应事件的压缩表征。
它不会再次抽帧，也不修改或消费原观测队列。M₁的无效切分不应调用此接口；标注中的原视频帧边界需要先映射到采样后的队列索引，不能直接混用。

下面只展示模块连接方式，不是训练脚本：

```python
import json
from pathlib import Path

from mem_lite.models.event_encoder import EventEncoder, encode_event_intervals
from mem_lite.models.event_queue import EventQueue

# segmenter 是已加载的 M1Segmenter；image_batch 是主视角图像的 processor 输出。
# image_batch 的张量需与 segmenter 位于同一设备。
frames = segmenter.extract_frame_features(image_batch["pixel_values"], image_batch["image_grid_thw"])
config = json.loads(Path("mem_lite/configs/event_encoder.json").read_text())
encoder = EventEncoder(input_dim=segmenter.visual_feature_dim, **config).to(frames[0].device)

# valid_intervals 来自 M1 有效切分，或已映射到观测队列的扰动标注区间。
encoded = encode_event_intervals(encoder, frames, valid_intervals)
queue = EventQueue(num_event_tokens=config["num_event_tokens"])
for event in encoded:
    queue.append(event)                 # 每个事件为 [32, hidden_dim]
first_event = queue.as_tokens(end=1)    # [32, hidden_dim]
all_events = queue.as_tokens()          # [事件数×32, hidden_dim]，作为 M2 输入
```

编码器在每次训练前向外创建并复用，不能为每个事件重新初始化。
每条视频使用独立 `EventQueue`；新视频开始时新建或 `clear()`。队列保存全部累计事件，没有固定容量淘汰，不保存旧 history。
同一事件的32个tokens共享事件序号的正余弦编码，仅在读取时相加，不修改底层事件表征，因此多次读取不会重复叠加位置编码。
`as_tokens(end=n)` 只提供前n个事件，后续入队不改变已读出的前缀。
空队列不生成伪造事件；调用方等首个有效事件入队后再触发 M₂。

### 梯度与下一阶段

ViT 特征提取使用 `no_grad`，可以缓存；事件编码器和队列不执行 `detach`。
事件 tokens 必须用当前编码器参数重新生成，不能把它们离线固化后继续训练事件编码器。
同一批训练样本的多个队列前缀可复用一次事件编码结果，汇总前缀损失后统一反向传播。
完成一次参数更新后，下一次前向应新建或清空训练队列并重新编码，不能跨优化步复用旧计算图。

M₂扰动片段的数据构造、加载与状态文本监督已接入下述第二阶段训练入口。
代码不会自动启动训练；离线检查中的梯度仅用于验证接口连通性，不是训练结果。

## M₂历史压缩器

`M2Compressor` 固定使用 **128个可学习query、4层交叉注意力＋FFN**，输出128个历史tokens，替代此前64个历史tokens的设计。
数量不随事件数改变，没有 query 池选取、门控或动态长度机制。
每层参数独立，复用事件编码器的注意力块实现，但不共享模型参数；使用相同的 Pre-LayerNorm、残差连接和最终输出 LayerNorm。

输入为 `EventQueue.as_tokens()` 生成的累计事件表征，已包含事件序号位置编码，M₂不重复添加。
每次前向从同一组可学习query开始，四层依次更新query，K/V始终来自本次传入的全部事件。
不接入任务指令、专家tokens或旧history，不裁剪最早的事件，不在模块内部缓存上次输出。
上层在新事件入队后调用它；何时调用的流式调度尚未实现。

```python
from mem_lite.models.m2_compressor import M2Compressor

config = json.loads(Path("mem_lite/configs/m2_compressor.json").read_text())
m2 = M2Compressor(input_dim=encoder.hidden_dim, **config).to(all_events.device)
history = m2([queue.as_tokens()])  # [1, 128, hidden_dim]

# 将不同长度的历史前缀作为一个批次；每一项独立生成128个历史tokens。
history_prefixes = m2([queue.as_tokens(end=1), queue.as_tokens()])
```

`forward(event_queues)` 的参数是张量列表，每个元素形状 `[累计事件数×32, input_dim]`。
不同视频或历史前缀可组成变长批次，内部 padding 使用注意力遮罩排除；输出形状 `[批大小, 128, hidden_dim]`。
默认内部维度512、8个注意力头、FFN扩展4倍和零dropout，输入维度通过 `encoder.hidden_dim` 取得。
这些宽度参数是可配置开发初值，固定128个query及4层结构直接定义在模块中。

事件编码器 → 事件队列 → M₂之间保留梯度，可通过下述冻结文本解码器的状态文本损失共同训练。
目前仅验证随机小张量前向、padding、顺序和梯度连接；尚未运行训练或验证实际预测效果。

## 独立状态文本解码器

`StateTextDecoder` 从本地**原始 Qwen3.5-0.8B 基座目录**加载独立的语言模型、词嵌入和原语言头，不加载 ViT，不共享 M₁实例或参数，不接入 M₁切分 LoRA。
使用 Transformers 的原生权重映射，将多模态检查点的语言部分加载为 `Qwen3_5ForCausalLM`；解码器全部参数冻结，始终保持 `eval()`。
内部可训练部分只有 `history_projection`：将 M₂输出维度线性映射到语言词嵌入维度。默认从512映射到0.8B的1024，实际目标维度从检查点配置读取。

```text
视觉特征 → 事件编码器 → 累计事件队列 → M₂
                                       ↓ 128个历史tokens
                                  可训练线性投影
                                       ↓ 连续embeddings
                              独立冻结的0.8B语言模型
                                       ↓
                                累计状态文本 CE
```

输入为固定指令“描述截至目前已完成的事件状态，仅输出状态描述。”和128个连续历史 embeddings。
使用原生聊天模板、关闭 thinking，将历史插在用户消息内；没有把历史离散化成文字或词表 token，也没有新增词表项。
不额外输入原任务指令、当前图像、专家或上一状态真值。
训练使用 teacher forcing：目标文本作为 assistant 答案接在提示后面，只计算答案及 EOS 的交叉熵，历史、提示和 padding 不计损失。
只对答案对应的隐藏位置计算词表 logits；序列过长时报错，不静默截断。长度上限默认2048，包含模板、128个历史位置及答案。

```python
from mem_lite.models.state_decoder import StateTextDecoder

# model_path 必须是原始本地基座，不是 M1 的 LoRA/训练检查点。
decoder = StateTextDecoder(model_path, history_dim=m2.hidden_dim).to(history_prefixes.device)

# 例：第一段完成后为“已拿起瓶子”，第二段完成后为“已完成1次撒调料”。
# 前缀及文本配对由数据构造层决定，解码器不推断或改写标注索引。
result = decoder(history_prefixes, ["已拿起调料瓶。", "已拿起调料瓶；已完成1次撒调料动作。"])
loss = result["loss_sum"] / result["text_tokens"]
```

`forward(history_tokens, state_descriptions)` 接收 `[B,128,history_dim]` 和B条状态描述，返回 CE 总和 `loss_sum` 与答案/EOS数量 `text_tokens`。
调用方对多个前缀、累积批次或多个 rank 的总损失与总 token 数统一归一化，避免按批均值再次平均造成偏差。
同一前向中多个前缀的损失汇总后再反向传播；冻结语言模型的前向**不能使用 `no_grad` 或 detach 历史**。
梯度经过冻结语言模型回传到历史投影、M₂及事件编码器；M₁仍由自己的切分损失更新，ViT保持冻结。
投影层默认FP32、语言基座默认BF16；本地微型检查使用FP32，正式CUDA/BF16联调尚未进行。

未来的 M₂训练检查点只需保存事件编码器、M₂及 `decoder.history_projection.state_dict()` 等可训练状态和基座路径，不重复保存冻结解码器权重。
部署进度预测器时移除这一文本监督分支。解码器同时提供训练损失接口和独立的`generate(history_tokens)`接口；
后者只输入历史及固定判断指令，不拼入真值文本，供下述M₂生成评测使用。尚未运行正式训练或真实权重评测。

## M₂数据构造与加载

数据从用户已经划定的训练视频构造，不自动切分验证集。以下命令仅为使用示例，本次没有对真实数据运行批量构造：

```bash
python -m mem_lite.datasets.prepare_m2 \
  --annotation /path/task/annotations/episode.annotations.json \
  --output results/mem_lite/m2/episode
```

主视角视频默认来自标注的 `video_file`，可用 `--video` 指定。复用固定10 FPS时钟、静止过滤和原始边界保留逻辑。
M₂对每个待编码区间按 `e=20%×min(自身及存在的相邻区间长度)` 扰动；先在 `[T-e,T+e]` 内均匀选取合法整数帧长度，再在该长度对应的合法整数起点中均匀采样。
实际窗口覆盖本区间至少80%，左右邻居各自最多20%，均允许恰好等于阈值。约束直接在整数帧上求解，不先采样再裁剪，也不通过重试改变分布。M₂不沿用M₁的2%离散步长或严格大于80%的判定。

每条视频默认生成8组扰动队列，由 `draws_per_video` 配置；随机性由seed固定。所有扰动在数据构造阶段完成，训练时不重新扰动。
若原标注有K段，一组样本只编码前K−1段，并构造K−1个累计前缀。前缀 `[p1,...,pi]` 固定监督为第i+1段的 `state_description`，不按扰动后的末帧重选文本。
初始状态不作为目标；末段不单独编码，但其描述和长度分别用于最后一个前缀的监督与相邻窗口约束。事件使用半开区间、原始帧轴；扰动片段之间可以有重叠或间隙，保持原事件顺序。

每个扰动事件在公共观测切片中补入自己的首帧和末帧，免静止过滤；不把其他样本的扰动边界混入自身。
输出集中于新目录：`frames/`、`samples.jsonl`、`annotation.json`、`observations.json`、`summary.json`，没有预计算或保存可训练的事件tokens。
`samples.jsonl` 每行保存一组事件帧清单、全部前缀标签、原始/扰动区间；不为每个前缀复制一份事件图片。

`M2Dataset` 在同一队列内按图片路径去重，返回图像列表、各事件的帧索引与前缀标签。
`M2Collator` 只处理视觉输入并保留每条队列的事件数，不把任务指令、状态描述或区间真值放入视觉编码器。
不同视频或扰动队列之间始终独立；边界之外的真值元信息仅留在清单供检查。

## 第二阶段联合训练

`Stage2Model` 运行两条独立监督分支：

- M₁使用已有切分样本，以JSON答案CE继续训练语言LoRA。
- 记忆分支使用扰动标注事件，经过冻结ViT、事件编码器、全部累计前缀的M₂压缩、独立冻结文本解码器，以状态文本CE训练事件编码器、M₂和历史投影。

记忆分支不调用M₁生成切分，避免在数据构造方案之外引入预测边界。两条分支共享同一个冻结ViT实例，文本损失不会更新M₁ LoRA。
同一M₂队列的重复图片只做一次ViT；ViT按 `vision_batch_size` 分批处理，避免一次送入整条视频的全部图片。各事件只编码一次，供全部前缀共同使用。
可训练事件tokens每次前向重新生成，不跨优化步缓存；每条队列独立创建，样本之间不回流历史。

**采样与损失默认值：** 一个epoch遍历全部M₂队列；每个队列配一条随机M₁样本，配对由 `seed/epoch/样本索引` 决定，可复现且每轮改变。
M₁配对与M₂队列可以来自不同视频，因为它们没有可训练的共享参数或逐样本对齐要求。M₁采用有放回抽样，单个epoch不保证遍历全部M₁样本；这是初版的联合采样配置。
`batch_size` 指每卡M₂队列数，也对应相同数量的M₁样本；一条队列的全部前缀参与监督，事件较多的视频贡献更多状态文本tokens。

目标函数为 `m1_loss_weight × 切分答案平均CE + memory_loss_weight × 状态答案平均CE`，默认权重均为1，分别可配置。
每条分支按当前完整梯度累积步中、全部rank的监督token总量归一化，避免两条分支文本长度不同改变损失权重。
实现先累积两条CE总和的梯度，再按两组互不重叠的可训练参数分别缩放；DDP的rank平均也在缩放中抵消。最后统一梯度裁剪并更新。
两条学习率默认均为1e-4，batch、像素数和梯度累积均为开发初值，尚未做显存/吞吐调优。

首次进入第二阶段必须提供第一阶段M₁检查点，继承其中的LoRA配置与权重；事件编码器、M₂和历史投影从初始化参数开始。
文本解码器始终来自原始基座。以下仅为未来运行示例，本次未执行：

```bash
torchrun --standalone --nproc_per_node=8 -m mem_lite.train_stage2 \
  --model-path /path/to/Qwen3.5-0.8B \
  --m1-checkpoint /path/to/m1/epoch-01.pt \
  --m1-sample-files /path/to/m1/episode_a /path/to/m1/episode_b \
  --m2-sample-files /path/to/m2/episode_a /path/to/m2/episode_b \
  --output-dir outputs/mem_lite/stage2
```

单卡改为 `python -m mem_lite.train_stage2`，其余参数相同；训练入口只支持CUDA BF16。
输出为 `config.json`、`train.jsonl`、周期覆盖的 `last.pt` 和每轮的 `epoch-XX.pt`。
日志分别记录切分CE、状态文本CE、加权总损失、答案token数及队列/前缀吞吐。
检查点保存M₁ LoRA、事件编码器、M₂、历史投影、优化器、下一个batch游标、总步数和各rank随机状态，不保存冻结基座/ViT/文本解码器。
仅在完整优化步后保存，支持每轮最后不足梯度累积步数的批次。

```bash
torchrun --standalone --nproc_per_node=8 -m mem_lite.train_stage2 \
  --resume outputs/mem_lite/stage2/last.pt --epochs 4
```

恢复使用检查点内的完整配置与参数，不再读取最初的M₁检查点，也不重新初始化记忆分支。
允许覆盖终止epoch/总步数和路径；须保持相同GPU数量、样本内容与顺序、batch及累积设置。基座路径变更应指向同一份原始权重。
数据loader每轮重建worker以同步epoch，配对不依赖worker随机数，跳过已完成batch后仍得到相同配对。
多卡使用等长DistributedSampler；样本数不能整除GPU数时补齐少量重复样本。
专家tokens、M₃、在线推理和验证评测均未接入本入口。

## M₁/M₂实际生成评测

两个入口均从完整的数据清单读取已构造样本，默认`--split validation`，不重新划分视频或抽取扰动。
应沿用训练时的清单；该验证集是任务内保留视频，不是独立测试集，也不是未见任务评测。
`--split train`只用于显式训练集诊断，输出清单会记录所用集合。
两者均采用贪心自回归、关闭thinking、使用独立KV缓存，不用teacher forcing，不从CE loss推算预测准确率。
每个输入窗口/队列独立处理，不回流上一个样本的状态。

未来有训练权重后，可按以下命令运行；本次只开发代码，没有执行这些正式评测命令：

```bash
python -m mem_lite.evaluate_m1 \
  --checkpoint outputs/mem_lite/m1/epoch-01.pt \
  --data-manifest results/mem_lite/dataset_v1/manifest.json \
  --output-dir results/mem_lite/evaluation/m1_epoch01 \
  --device cuda:0 --batch-size 1

python -m mem_lite.evaluate_m2 \
  --checkpoint outputs/mem_lite/stage2/epoch-01.pt \
  --data-manifest results/mem_lite/dataset_v1/manifest.json \
  --output-dir results/mem_lite/evaluation/m2_epoch01 \
  --device cuda:0 --batch-size 1 --prefix-batch-size 4
```

M₁接受第一阶段或第二阶段检查点；M₂需要第二阶段的事件编码器、压缩器和历史投影权重。
基座及图像配置默认来自检查点，目录迁移后可用`--model-path`指向同一份原始0.8B权重。
入口当前为单进程，支持CUDA BF16和用于离线接口检查的CPU FP32，未做多卡分发或正式吞吐调优。
`--batch-size`分别表示M₁窗口数、M₂完整事件队列数；`--prefix-batch-size`控制M₂解码时的前缀批次大小。
`--limit N`只取前N个窗口/队列，结果标记`partial`，不冒充全量结果。
`--max-new-tokens`默认M₁为96、M₂为256；超出预算但未遇到结束符的输出标记`truncated`，仍进入评分。

每次使用新的输出目录，保留四个文件：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 权重/样本清单SHA-256、基座路径、集合、生成配置、完整或部分评测及成功/失败状态 |
| `predictions.jsonl` | 逐样本原始生成文本、token IDs、真值、来源和各项误差 |
| `errors.jsonl` | M₁未完全匹配目标JSON、M₂归一化文本不一致的样本，保留完整原文 |
| `metrics.json` | 总体指标及按任务/视频/M₁ case或M₂前缀长度分组的指标 |

**M₁：** 分别统计JSON可解析率、完整格式合法率、valid准确率/精确率/召回率/F1与混淆矩阵、完整答案一致率、区间IoU、起止边界MAE（原帧数及秒）。
`valid`字段判断与区间格式分开计分；例如`valid=1`但边界越界，可以正确判断有效性，但格式不合法、IoU为0。
模型输出的`[start_index,end_index)`映射成`[起始观测原帧号, 最后选中观测原帧号+1)`，不映射到下一张稀疏帧。
IoU均值及IoU≥0.5召回率的分母包括**全部真值有效窗口**，漏检、无法解析或边界非法计0。
边界MAE只统计真值有效且预测有效、边界合法的窗口，同时给出`boundary_evaluated_samples`，需与IoU及召回率一起阅读。
输出不做JSON修复、不纠正边界顺序、不去除代码围栏后再重试。

**M₂：** 使用已构造的扰动事件队列，由冻结ViT、事件编码器、M₂生成每个累计前缀的128个历史tokens，再通过独立文本解码器实际生成。
真值状态仅参与评分；历史编码及生成不接收真值文本。不使用M₁预测边界，因此衡量记忆分支能力，尚未覆盖在线切分错误的传播。
每个前缀是一条评测样本，同视频的不同扰动及不同前缀都会计入；总准确率按前缀汇总，不按视频等权平均。

| 指标 | 计算含义 |
|---|---|
| 原文/归一化一致率 | 原文完全相同；或仅进行Unicode NFKC与空白归一化后相同 |
| 字符错误率 CER | 总字符编辑距离 ÷ 总真值字符数；可能大于1 |
| 状态子句指标 | 按分号、中文句号及换行分割，计算子句有序一致率、集合一致率及集合micro-F1 |
| 仅子句重排样本数 | 子句及重复次数相同，但出现顺序不同 |
| 次数序列一致率 | 真值含显式“数字＋次/遍”的前缀中，按出现顺序比较数值；支持常见中文整数，漏项/多项计错 |
| 条件次数MAE | 仅对预测和真值次数项数相同的前缀逐项计算绝对差，同时列出可比较前缀/次数项数量 |

这些是可复核的**词面指标**，同义表达可能被计错；次数比较不单独判断数值属于哪个对象或动作。
子句顺序与机器人实际动作时序不能等同，不能据此声称模型掌握了语义顺序推理。
无适用样本的指标保存为`null`，不记为0或100%；原文可用于人工复核。

## M₃进度预测模型

`M3ProgressPredictor` 使用独立的 Qwen3.5-4B 基座及其原生 ViT。它不复用 M₁的0.8B视觉编码器，
不增加视觉适配模块，也不接专家参考或状态文本解码器。

- 冻结：M₃基座、原生ViT及视觉合并层、词嵌入、原语言输出头。
- 训练：语言Transformer各层Linear的LoRA、独立历史线性投影、线性进度分类头。
- 第三阶段上游M₁、事件编码器和M₂保持冻结；M₃在接收历史tokens时截断上游梯度。
  上游模块不由本类创建或调用；`FrozenEventMemory`负责其`eval()`、`requires_grad_(False)`和`no_grad()`，
  第三阶段入口读取预先生成的冻结历史缓存，不在每个训练batch内重复运行记忆分支。

### 输入与历史接入

`M3Collator` 接收每条样本的：

- `task_prompt`：任务指令。
- `images`：同一时刻的3张PIL图像，固定视角顺序由调用方保持；分别经过原生图像处理器及ViT。
- `history_tokens`：`[128, history_dim]`张量，默认内部维度512；首个事件出现前传`None`。

排列为：`任务指令 → 历史embeddings（如有） → 当前三视角 → 固定进度查询提示`，使用原生chat template，关闭thinking。
历史经过M₃自己的线性投影进入4B语言隐藏维度；不转换成描述文字，不复用M₂文本解码器的投影。
输入不包含PT真值、阶段真值或目标状态描述。

实现上在第一个图像块前预留128个位置，使用原词表pad ID暂存，模型前向时替换为连续历史embeddings；
这些历史位置的attention mask为1、模态类型为文本，不扩充词表。
在完整序列上重算原生3D RoPE，保持三视角空间位置编码，避免直接传`inputs_embeds`时丢失图像位置信息。

有历史和无历史样本可以混合成批。无历史样本在语言序列中不预留任何历史位置；
批次历史张量中的补零仅用于对齐，不被投影或输入该样本。全批无历史时`history_tokens=None`。
序列使用左侧padding；超过长度上限报错，不截断历史或三视角图像。

### 进度头与监督

读取最后一个有效输入token的最终隐藏状态，线性头输出101个取值点`0, 0.01, ..., 1`的logits。
一次前向给出分布及其概率加权期望`pt`，不调用`generate`或原词表输出头。
例如真值`0.2375`对应`0.23`位置权重`0.25`、`0.24`位置权重`0.75`；端点0、1使用单点标签。

`forward(batch, target_pt=None)` 返回 `pt`、`bin_logits`、`bin_probs`。
传入`[batch_size]`、范围`[0,1]`的`target_pt`时，另返回软标签交叉熵`loss`（样本均值）、`loss_sum`和`num_samples`。
PT仅用于损失，不进入语言输入；不施加进度单调约束，允许相邻时刻的预测回落。
bin概率、连续PT和损失保留FP32，即使语言前向使用BF16 autocast。

以下只展示模型连接方式，不是已执行的训练；逐帧PT文件见下文：

```python
import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from mem_lite.datasets.m3_inputs import M3Collator
from mem_lite.models.m3_predictor import M3ProgressPredictor

model_path = "/path/to/Qwen3.5-4B"
config = json.loads(Path("mem_lite/configs/m3_model.json").read_text())
model = M3ProgressPredictor(model_path, **config).cuda()
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, max_pixels=65536)
collator = M3Collator(processor, history_dim=config["history_dim"])
# view0/view1/view2为同一时刻三视角PIL图像，history来自冻结M₂；无历史时为None。
batch = collator([{"task_prompt": "点击按钮三次", "images": [view0, view1, view2],
                   "history_tokens": history}])
batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    result = model(batch)
```

本次检查使用离线随机微型Qwen权重，没有启动训练或下载新权重；未对M₃进行真实4B/CUDA联调。

## 第三阶段数据接口与训练入口（代码预备）

**当前边界：先训练M₁、M₂，再确定真实M₃样本的历史生成方式。PT已采用事件等权、段内视觉delta分配，
以逐帧CSV保存；历史事件切分来源仍未定稿，没有运行批量历史缓存生成。**

`M3Dataset`读取已经准备好的目录，每条视频独立保存：

```text
samples.jsonl    每行包含 frame_index、time_seconds、images、history_index 和 pt
summary.json     格式版本、任务指令、历史维度、历史来源和history.pt的SHA-256
history.pt       history_tokens [K,128,history_dim] 与 available_from_frame [K]
frames/          当前帧的三视角图像，路径由samples.jsonl明确指定
```

`images`按`cam_high`、`cam_left_wrist`、`cam_right_wrist`命名，加载时使用这一固定顺序，不能用主视角复制补齐缺失视角。
`history_index`从0开始，指向累计前缀的历史张量；首个事件可用前为`null`。
调用方在生成缓存时负责记录每个前缀的真实可用帧号，不能把未来画面编码后提前提供给当前帧。
Dataset只允许选择当前帧已经可用的最新前缀，拒绝未来或过期前缀；每个worker最多缓存4条视频的历史张量。
没有任何可用历史的视频使用`[0,128,history_dim]`空缓存，不能用伪造的零历史代替。

`summary.json`的`format_version`为`mem-lite-m3-v1`；`task_prompt`为任务指令；`history_dim`需匹配模型；
`history_sha256`对应该视频的缓存文件；`history_source`应记录共同的第二阶段权重摘要和历史构造规则。
同一次训练不混用不同来源的缓存。样本、任务指令、历史来源和缓存内容组成数据签名，续训时核对，支持整体迁移路径。
图像文件需要保持不变；入口不会每次启动都对所有图片重新计算哈希。

PT为每行明确提供的有限`[0,1]`数值，允许回落，缺失不推算，越界不截断。
`Stage3Collator`分别返回`inputs`和`target_pt`；PT及状态描述不拼入模型提示。
上述格式是数据加载接口，不表示真实逐帧PT或历史缓存已生成。

`FrozenEventMemory`只接受第二阶段检查点，加载M₁、事件编码器和M₂并全部冻结；不加载文本解码器。
其前向接收一条视频的主视角视觉输入及有序事件帧索引，只调用M₁的ViT特征提取；不调用M₁语言切分前向。
事件编码一次后逐个入队，每次用全部事件重新生成128个历史tokens，不回流旧历史。
此模块没有自行决定事件来自标注还是预测，真实事件清单与可用时刻由后续数据构造流程提供。

训练入口仅加载M₃及冻结历史缓存，只更新语言LoRA、M₃自己的历史投影和进度头。
软标签CE按整个梯度累积步、所有rank的实际样本数归一化，包含每轮末尾不足累积步数的批次。
DDP启用`find_unused_parameters=True`，处理某个rank或整个微批次没有历史、历史投影不参与计算的情况。
日志包含训练CE、PT的MAE/RMSE/偏差（百分点）、吞吐；这些是训练样本统计，不是验证集指标。

以下是未来数据与第二阶段权重准备好后的命令示例，本次没有执行：

```bash
python -m mem_lite.train_stage3 \
  --model-path /path/to/Qwen3.5-4B \
  --sample-files /path/to/prepared_m3/episode_a /path/to/prepared_m3/episode_b \
  --output-dir outputs/mem_lite/stage3

torchrun --standalone --nproc_per_node=8 -m mem_lite.train_stage3 \
  --resume outputs/mem_lite/stage3/last.pt --epochs 4
```

入口仅支持CUDA BF16，单卡与多卡使用相同配置。默认batch和学习率是开发初值，未做吞吐调优。
检查点保存M₃ LoRA、历史投影、进度头、优化器、下一个batch位置、总步数、配置和各rank随机状态；
不重复保存冻结基座或上游模型。续训保留GPU数量、batch/累积设置和数据内容，只允许命令行覆盖路径及终止轮数/步数。
没有验证集自动划分、评测、M₁在线切分或自动启动训练流程。

## 逐帧PT标签：事件等权、段内按视觉变化分配

标签生成方法现已确定，第三阶段历史构造方式仍待后续确认。
`prepare_progress.py`使用冻结Qwen3.5视觉编码器，读取merger后的全部空间tokens并在FP32中平均池化；
相邻原始帧的向量L2距离为delta。只加载ViT和merger，不加载语言权重或LoRA，不需要已训练的M₁/M₂。
原视频25/30FPS的全部帧都参与计算，包括静止帧；不使用10FPS队列采样，不添加epsilon或改成时间线性进度。

以整条视频为单位，JSON中的K个区间全部参与计算，每段等权占1/K，不根据`state_description`筛除区间。
初始“无阶段性事件”区间和最后的收手、复位区间同样计入；段内按累计delta占该段总delta的比例分配。
这一定义表示整段标注流程的推进，不单独推断任务成功，也不产生失败重做/撤销时的进度回退。

**边界约定：** 视频共N帧，区间[a,b)的右锚点为c=min(b,N−1)。
该段累计delta[a+1]到delta[c]，即图像a→a+1直至c−1→c的变化。
第k段（从0编号）的PT[a]=k/K、PT[c]=(k+1)/K；delta[a]属于前一段，不重复计算。
普通段的右锚点位于下一段首帧，最后一段的右锚点位于视频最后一帧，不读取不存在的第N帧。
例如739帧、8段的视频，第702帧进入最后一段时PT=7/8，第738帧达到1。
首帧delta与PT均为0。任何段的delta总和≤1e-8会报错要求复核（包括无可见变化的末段），不自动生成替代标签。

每条视频的新输出目录只生成：

- `episode_XXXXXX.progress.csv`：`frame_index,delta,pt`，原视频一帧一行，连续PT保留在[0,1]。
- `episode_XXXXXX.metadata.json`：视频/标注/实际视觉权重摘要、模型和精度、原始时间戳、事件完成锚点及计算规则。

运行示例（冻结视觉推理，不是训练）：

```bash
python -m mem_lite.datasets.prepare_progress \
  --annotation /path/MemLong/task/annotations/episode.annotations.json \
  --video /path/MemLong/task/episode.mp4 \
  --model-path /path/to/Qwen3.5-4B \
  --output-dir results/mem_lite/progress_preview/episode \
  --device cuda --batch-size 8
```

当前使用Qwen3.5-4B，区别于旧脚本的Qwen3-VL-4B；保留相同的平均池化/L2定义，但两种编码器的delta不可直接混用。
CSV既保留成本较高的delta，也保存可直接监督M₃的PT，后续调整事件权重时可复用delta重新计算。

批量生成时只扫描`任务目录/annotations/*.annotations.json`对应的主视角视频，不处理未标注视频或重复处理腕部视角。
模型只加载一次，各视频结果保存到`输出目录/任务名/视频名/`，CSV及元数据格式与单视频相同：

```bash
python -m mem_lite.datasets.prepare_progress_batch \
  --dataset-root /path/MemLong \
  --model-path /path/to/Qwen3.5-4B \
  --output-dir results/mem_lite/progress_labels/run \
  --device cuda --batch-size 8
```

批次根目录的`manifest.json`记录输入清单，`index.jsonl`索引成功结果，`status.json`和`result.json`记录运行状态与失败原因。
单条视频失败后保留错误记录并继续下一条，不自动替换异常标签；每次运行使用新输出目录，不覆盖已有批次。

## 检查

```bash
python -m pip install pytest
python -m pytest mem_lite/tests -q
```

仅检查数据构造时运行 `mem_lite/tests/test_m1_data.py`，无需安装训练依赖。
训练接口检查使用随机微型 Qwen3.5 结构和模拟输入，不下载模型，不执行训练循环或优化器更新。
事件模块检查使用随机小张量，并以一次反向计算验证四层编码器、输入投影、query与事件队列之间的梯度连接。
文本解码器检查在 pytest 临时目录生成随机微型多模态检查点和 tokenizer，核对原始语言权重加载、与M₁独立、答案/EOS掩码、变长批次损失及跨冻结解码器的梯度。
第二阶段检查使用合成视频与原生processor的微型模型前向，验证真实图像批次、两分支梯度范围、损失缩放及可训练状态的保存恢复；不调用训练循环或执行优化器更新。
生成评测检查使用随机微型Qwen及合成视频，实际执行短程自回归，核对左右padding/缓存、无标签生成、验证集读取和结果落盘；这些输出只是测试临时产物。
这些检查只验证代码连接，不代表真实0.8B已经理解历史tokens或具备状态预测效果。
