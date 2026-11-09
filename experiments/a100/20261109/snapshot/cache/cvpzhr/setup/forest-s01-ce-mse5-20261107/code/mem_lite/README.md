# StreamForest 视觉记忆与进度预测

本分支已移除 M1 事件切分、事件编码器、M2 压缩、状态文本解码及对应训练和评估入口。
新的记忆链路为：冻结 Qwen3.5 ViT/merger → 带时空位置的视觉 tokens → StreamForest。
记忆核心没有可训练参数，不需要旧阶段的检查点。

## 文件布局

```text
mem_lite/
  memory/
    tome.py                  ToMe 加权 token 合并
    manager.py               当前窗口、短期窗口和长期事件记忆
    vision.py                仅加载 Qwen 视觉权重、提取 tokens、添加时空位置
    stream.py                跨调用保留状态、自动按视频/任务重置
    LICENSE.StreamForest     上游许可证
  configs/streamforest.json  记忆窗口与预算
  models/m3_predictor.py     三视角 + 有效历史 tokens 的进度预测
  datasets/prepare_stream.py 同步视频与 PT 标签构造训练样本
  datasets/m3_dataset.py    按需读取历史快照
  datasets/m3_inputs.py     固定容量历史张量和有效长度 mask
  train_stage3.py           LoRA、历史投影与进度头训练
  runtime.py                检查点加载、持续在线预测
  evaluate.py               缓存/在线评估、延迟与显存统计
  tests/                     记忆核心、真实微型 Qwen 与进度回归测试
```

## 使用记忆模块

```python
import json
from pathlib import Path

from mem_lite.memory import MemoryManager
from mem_lite.memory.vision import QwenFrameEncoder, load_vision_encoder

visual, processor, _ = load_vision_encoder("/path/to/Qwen3.5", "cuda")
encoder = QwenFrameEncoder(visual)
config = json.loads(Path("mem_lite/configs/streamforest.json").read_text())
memory = MemoryManager(encoder.hidden_size, **config)

# images 为同一视频主视角按时间排列的 PIL 图像。
# 已有 VLM 时也可以 QwenFrameEncoder(vlm.model.visual)，共享冻结视觉模块。
inputs = processor(images=images, return_tensors="pt", max_pixels=262144)
frames = encoder(inputs["pixel_values"], inputs["image_grid_thw"], frame_positions=frame_indices)
for frame in frames:
    memory.update(frame)
history_tokens = memory.get_memory_tokens()  # [N, encoder.hidden_size]；空历史为 None
```

`frame_positions` 使用绝对帧位置；分批处理时延续位置，不从零重计。
每个 memory 对应一条单视角视频流，`reset()` 清空它；多视角不要交替塞进同一事件队列。
`update()` 每次都执行短期迁移、事件切分和长期压缩，读取结果不再修改状态。
实际在线调用使用下面的 `StreamingProgressPredictor`，训练缓存和在线推理共用 `StreamMemory`。

默认采用上游 `generate_online` 示例的窗口预算：当前 1 帧最多 64 tokens，
短期窗口 6 帧、切分缓冲 4 帧，每帧最多 16 tokens，长期最多 800 tokens。
整理后短期最多 9 帧，因此总上限为 `64 + 9×16 + 800 = 1008`；
`memory.token_budget` 返回这个上限。预算不足时继续合并，即使只剩一个长期事件也不会超额。
实际输出可以小于上限，不补造 tokens，也不会将低分辨率输入扩增至预算长度。

## 与上游的适配差异

实现源自 [MCG-NJU/StreamForest](https://github.com/MCG-NJU/StreamForest)，
固定参考提交 `f9d14670ff451ff5865638e17c9958e99bf2fec5` 的
[`memory_manager.py`](https://github.com/MCG-NJU/StreamForest/blob/f9d14670ff451ff5865638e17c9958e99bf2fec5/llava/model/multimodal_projector/memory_manager.py)。
时空位置参考其 `projector_FSTW_PEMF.py`，预算参考 `llava_qwen.py`。
保留 ToMe 的加权合并、基于相似度的事件切分，以及内容/时间/合并次数三项合并代价。
移植文件已精简并修改，许可证见 `memory/LICENSE.StreamForest`。

- 使用 Qwen 原生 merger 之后的视觉特征，输出维度由 `vision_config.out_hidden_size` 决定；不使用 SigLIP 权重或原项目的 MLP。
- Qwen 支持矩形、可变大小的图像网格。事件边界相似度用每帧特征均值计算，记忆本身仍保留空间 tokens。
- 合并匹配默认使用完整特征向量（`num_attention_heads=1`），不将 merger 输出强行解释为原视觉注意力头。
- 长期事件按实际 token 数折半，不再强制固定每帧 token 数的整倍数；时间信息随合并更新。
- 显式添加时间/行/列正余弦位置，时间占通道的四分之一；不沿用上游的 512 帧循环位置。
- 归一化和合并累加使用 FP32，输出维持输入 dtype。

## 因果关系与固定容量输入

记忆仅接收主视角 `cam_high`，当前观测保持 `cam_high / cam_left_wrist / cam_right_wrist` 三视角顺序。
预测第 t 帧时，历史只包含之前已观测的帧；当前帧更新后的记忆只供下一次预测使用。
视频首帧历史为空，切换 `episode_id` 或任务指令会自动清空状态；相同视频重播需显式 `reset()`。

M3 接收固定容量的 `history_tokens: [batch, capacity, D]` 和
`history_valid_mask: [batch, capacity]`。不足部分补零；仅有效 tokens 经投影后插入 VLM。
`history_mask` 标识它们在完整语言/图像序列中的位置，两种 mask 的有效数量一致。
因此 padding 不占用历史语义位置，也不影响 Qwen 的多模态位置或线性注意力。
D 从缓存元数据和 Qwen 原生视觉输出确定，不再写死为 512；默认容量为 1008。

## 构造训练样本

输入为已经按帧同步、帧率和帧数相同的三视角视频，以及原有进度标签 CSV
（至少包含 `frame_index,pt`，也接受原来的 `frame_index,delta,pt`）。

```bash
python -m mem_lite.datasets.prepare_stream \
  --videos /data/episode_cam_high.mp4 /data/episode_cam_left_wrist.mp4 /data/episode_cam_right_wrist.mp4 \
  --progress-csv /data/episode.progress.csv \
  --task-prompt '将物品依次放入抽屉' \
  --model-path /models/Qwen3.5-4B \
  --output-dir /data/streamforest/train/episode \
  --sample-fps 5 --device cuda
```

默认按 **5 FPS** 采样，每 0.2 秒一次同步三视角观测：25 FPS 视频每 5 帧取一帧，
30 FPS 视频每 6 帧取一帧。不能整除时，取达到采样时刻的第一帧；源视频不足 5 FPS 时保留所有帧。
采样同时决定记忆更新和预测频率，跳过的帧不进入记忆；在线部署应沿用训练频率。
`--frame-stride` 可显式改用固定帧间隔，与 `--sample-fps` 互斥。
窗口大小以输入观测帧为单位。`--memory-config` 可替换预算配置，
`--max-pixels / --min-pixels` 控制记忆图像分辨率；这些设置随缓存和训练检查点保存。

每个输出目录包含三视角 PNG、`samples.jsonl`、`summary.json` 以及逐观测的 `history/*.pt`。
快照保留编码器输出 dtype，仅保存有效 tokens，不写满容量 padding。
每个快照记录其最后观测帧；标签不会传入视觉编码器或记忆管理器。
构造和加载都不把整条视频的所有历史快照留在内存，但长视频的磁盘占用仍随样本数增长。
训练与验证按完整视频分开准备。

## 训练

```bash
python -m mem_lite.train_stage3 \
  --model-path /models/Qwen3.5-4B \
  --sample-files /data/streamforest/train/episode \
  --output-dir /outputs/streamforest

# 多卡使用相同参数
# torchrun --standalone --nproc_per_node=8 -m mem_lite.train_stage3 ...
```

`configs/stage3_train.json` 中 `history_capacity` 和 `model.history_dim` 为 null，
训练时从样本解析并写入检查点。只优化语言 LoRA、历史投影和进度头；视觉模块及记忆算法冻结。
断点续训仍使用 `--resume /outputs/streamforest/last.pt`。
`save_every_steps` 控制步数快照，设为 null 可关闭；`save_every_epochs` 默认 1。
例如设为 3 时，只在第 3、6、9… 个 epoch 保存 epoch 存档，并更新 `last.pt`。
新检查点格式为 `mem-lite-streamforest-v1`，不接受旧 M1/M2 或旧固定 128 tokens 的进度检查点。
训练入口使用 CUDA BF16，支持 DDP、梯度累积和断点续训。
预测模型内部统一训练与推理的 autocast 策略；记忆编码和 FP32 合并匹配不受外层 autocast 影响。
训练启动及检查点加载时，会核对缓存记录的视觉权重摘要，避免混用维度相同但权重不同的基座。

## 持续在线预测

```python
from mem_lite.runtime import StreamingProgressPredictor, load_checkpoint

model, processor, config = load_checkpoint('/outputs/streamforest/last.pt', device='cuda')
predictor = StreamingProgressPredictor(
    model, processor, config['history_source'], config['max_sequence_length'],
)
result = predictor(
    [cam_high, cam_left_wrist, cam_right_wrist],  # 当前时刻的 PIL 图像
    frame_index=frame_index, episode_id='task/episode', task_prompt='将物品依次放入抽屉',
)
pt = result['pt'][0].item()
# 下一次继续调用同一个 predictor；主动重播同一视频时 predictor.reset()。
```

调用方按检查点 `history_source['sample_fps']`（默认 5 FPS）提供同步观测，
并传入原始视频帧号；`predictor` 每次调用都会更新记忆，不在内部限频。
在线评估直接沿用缓存样本的帧号和采样频率。
历史编码器与 M3 共享同一个冻结 ViT，不加载第二份语言模型。
当前实现另编码一次主视角来更新记忆；在线延迟统计包含这次编码，不重算全部历史帧。

## 评估与检查

```bash
# 缓存评估
python -m mem_lite.evaluate \
  --checkpoint /outputs/streamforest/last.pt \
  --sample-files /data/streamforest/validation/episode \
  --output-dir /outputs/eval-cached

# 同一验证视频在线重建记忆，不读取历史快照
python -m mem_lite.evaluate \
  --checkpoint /outputs/streamforest/last.pt \
  --sample-files /data/streamforest/validation/episode \
  --output-dir /outputs/eval-online --online

# 加 --without-history：对同一个检查点做移除历史的消融
python -m pytest mem_lite/tests -q
```

评估输出 `predictions.jsonl` 和 `metrics.json`：MAE/RMSE/bias（进度百分点）、
首帧延迟、后续帧平均/P50/P95 延迟以及 CUDA 显存峰值。
延迟不含读图、解码和读缓存 IO；在线模式包含预处理、主视角编码、记忆更新及 VLM 前向。
CPU 评估的 CUDA 显存字段为 null。无历史消融使用同一检查点，不代表单独训练的无历史基线。

进度标签生成仍使用 `mem_lite.datasets.prepare_progress` 和 `prepare_progress_batch`，计算规则保持原样。
依赖沿用 `requirements-data.txt` 和 `requirements-train.txt`。
测试覆盖预算、因果性、自动重置、变长有效历史、padding 隔离、真实视频输入、
FP32/BF16 缓存与在线一致性、未来帧替换、微型 Qwen 优化器更新及检查点重载；无需下载模型。
这些离线测试不代表真实任务准确率或 GPU 实时性能。

## 插螺钉恢复数据

`prepare_recovery` 读取 `30FPS/`、`left_wrist/`、`right_wrist/`、`splitlable/` 和
`delta/`，强制排除 `file-019`。它复用 `qwen_training.delta_to_progress.episode_to_progress`，
按 `nowid/restid` 扣除失败后需要重做的任务贡献；不使用事件等权的单调进度算法。
每轮单独归一化，JSON 中包含末帧的区间转换为 `[start_frame,end_frame)`，CSV 仍保留原视频帧号。
片头准备、轮间空白和片尾空闲不进入单轮训练样本。

```bash
# CPU：三视角帧数/帧率/起始时间、delta 和标注校验；按原始视频划分。
python -m mem_lite.datasets.prepare_recovery index \
  --dataset-root /root/autodl-tmp/datasets/insert-screw-recover \
  --output-dir /root/autodl-tmp/streamforest-data/insert-screw-recover \
  --validation-videos 4 --seed 42

# 冻结视觉推理（不更新模型参数）：一次加载视觉编码器，每轮独立构建历史。
python -m mem_lite.datasets.prepare_recovery cache \
  --prepared-dir /root/autodl-tmp/streamforest-data/insert-screw-recover \
  --model-path /root/autodl-tmp/models/Qwen3.5-4B \
  --cache-dir /root/autodl-tmp/streamforest-cache/insert-screw-recover \
  --sample-fps 5 --device cuda

# 缓存完成后再训练；验证集不会被写入训练配置。
python -m mem_lite.train_stage3 \
  --config /root/autodl-tmp/streamforest-cache/insert-screw-recover/train.config.json
```

当前 21 个可用原始视频分成 17 个训练视频（85 轮）和 4 个验证视频（20 轮）；
固定验证视频为 `file-000/003/008/021`。同一视频的所有轮都留在同一划分。
索引目录保存 `manifest.json`、`train.jsonl`、`validation.jsonl` 和逐轮 `progress.csv`。
缓存以每轮首帧为空历史开始，无法读取该轮之外的帧；按 5 FPS 采样并额外保留每轮末帧，
确保完成状态也被监督。原三视角视频不会被重编码或裁剪。
缓存目录最后生成 `train.config.json` 与训练/验证各自的 `*.sample_files.json`。
